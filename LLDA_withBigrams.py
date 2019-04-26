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
from collections import Counter
 
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

    #def inference_query(self, test_doc):
	#t1 = time.time()	
        #V = len(self.vocas)
	
	#for t, z in zip(testdoc, test_z_n):
	    #test_n_z[z] += 1  
        
	#for n in range(len(test_doc)):
        #    t = testdoc[n]
	#    if (t< V):
        #        z = test_z_n[n]
        #    	    
        #        denom_a = test_n_z.sum() + self.K * self.alpha
		# Commented by praveen 
                #denom_b = self.n_z_t.sum(axis=1) + V * self.beta
            
	#	p_z = testlabel * (self.n_z_t[:, t] + self.beta) / denom_b * (test_n_z + self.alpha) / denom_a
        #        new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
        # 	 newvalue = p_z[new_z]
	#    else:
	#	z = test_z_n[n]
	#	
        #   	denom_a = test_n_z.sum() + self.K * self.alpha
	#	# commented by praveen no need to compute everytime
        #   	#denom_b = self.n_z_t.sum(axis=1) + V * self.beta

        #    	p_z = testlabel * (self.beta) / denom_b * (test_n_z + self.alpha) / denom_a
		
        #    	new_z = numpy.random.multinomial(1, p_z / p_z.sum()).argmax()
	#	newvalue = p_z[new_z]
	#    test_z_n[n] = new_z
	#t2 = time.time()
	#print "timetaken :" , t2-t1
	#return test_n_z 

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

corpus =  corpus_[0:200]
labels_ = labels_r_[0:200]

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

print len(reuters_corpus)
reuters_corpus_doc = []
bigram_list = []
bigramslist = []
for doc in reuters_corpus:
    for i in range(len(doc) - 1):
	bigrams = doc[i] + ' ' + doc[i + 1]
	bigram_list.append(bigrams)
    corp_doc = doc + bigram_list
    reuters_corpus_doc.append(corp_doc)
    bigramslist.append(bigram_list)
    bigram_list = []	
    corp_doc = []

corpus_bigrams = list(itertools.chain(*bigramslist))

counts_ = Counter(corpus_bigrams)
print len(counts_)

rejectwords=([word for word,v in counts_.items() if v == 1])
#print rejectwords
print len(rejectwords)

reuters_corpus_unibigrams =  []
reuters_term = []
for document in reuters_corpus_doc:
    for term in document:
	if term not in rejectwords:
	    reuters_term.append(term)
    reuters_corpus_unibigrams.append(reuters_term)
    reuters_term = []
print len(reuters_corpus_unibigrams)

llda = LLDA(options.K, options.alpha, options.beta)

llda.set_corpus(labelset, reuters_corpus_unibigrams, labels)

print "M=%d, V=%d, L=%d, K=%d" % (len(reuters_corpus), len(llda.vocas), len(labelset), options.K)

for i in range(options.iteration):
    sys.stderr.write("-- %d : %.4f\n" % (i, llda.perplexity()))
    llda.inference()
print "perplexity : %.4f" % llda.perplexity()

phi = llda.phi()

#for k, label in enumerate(labelset):
#    print "\n-- label %d : %s" % (k, label)
#    for w in numpy.argsort(-phi[k])[:50]:
#        print "%s: %.4f" % (llda.vocas[w], phi[k,w])

#  New Code for topic representation in terms of frequency patterns
words = []
documents = []
topics = []
topic_patterns_list = []
patters_list = []
for top in range(len(labelset)):
    for m, doc in zip(range(len(reuters_corpus)), reuters_corpus):
	for n in range(len(doc)):
            if ( top == llda.z_m_n[m][n]):
                words.append(reuters_corpus_unibigrams[m][n])
	documents.append(words)
	words = []
    patters_list = list(itertools.chain(*documents))
    topic_patterns_list.append(patters_list)
    documents = []
    patters_list = []

#g = []
#patterns = []
#topic_patterns = []
#topic_patterns_list = []
#tpes = []
#c = 0
#for tap in topics:
#    patterns_list = list(itertools.chain(*tap))
#    for tp in tap:
#	for i in range(len(tp)-1): 
#	    tpc = tp[i] + ' ' + tp[i + 1]
#	    g.append(tpc)
#	tpes.append(g)
#	g = []
    
#    patters_list = list(itertools.chain(*tpes))
#    tpes = []
#    ptrs_list = patterns_list + patters_list
#    patterns_list = []
#    patters_list = []
    #counts = Counter(ptrs_list) 
    #reject_words=([word for word,v in counts.items() if v == 0])
    #for filter_list in ptrs_list:
    #    if filter_list not in reject_words:
    #	    topic_patterns.append(filter_list)
    #topic_patterns_list.append(topic_patterns)
    #topic_patterns = []
#    topic_patterns_list.append(ptrs_list)
#    ptrs_list = []

dictionary = corpora.Dictionary(topic_patterns_list)

corpus = [dictionary.doc2bow(text) for text in topic_patterns_list]

# creating the transformation
tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model

corpus_tfidf = tfidf[corpus]

dic = {}
list_of_dic = []
for doc in corpus_tfidf:
    for elem in doc:
	dic[elem[0]] = elem[1]
    list_of_dic.append(dic)
    dic = {}

dict_id = dictionary.token2id

#print len(list_of_dic)
#print(dictionary)
#print len(dict_id)
#print dict_id


testing_unigrams = reuters_corpus[75]

print testing_unigrams
print labels[75]

testing_bigrams = []
for test in range(len(testing_unigrams)-1):
    t_c = testing_unigrams[test] + ' ' + testing_unigrams[test + 1]
    testing_bigrams.append(t_c)

test_query = testing_unigrams + testing_bigrams
print test_query
print len(test_query)
test_res = numpy.zeros(len(labelset), dtype=float)
res_ind = 0
testlabels = []
for test_word in test_query:
    if test_word in dict_id.keys():
        word_id = dict_id[test_word]
        #print test_word
        for dic in list_of_dic:
	    if word_id in dic.keys():
	        num = dic[word_id]
                denom = sum(dic.values())
	        val = num/denom
	    else:
	        val = 0
            test_res[res_ind] = val
	    res_ind = res_ind + 1
        res_index = test_res.argmax()
        testlabels.append(labelset[res_index])
        res_ind = 0
        test_res = numpy.zeros(len(labelset), dtype=float)
print testlabels
print len(testlabels)
#print labelset
results = Counter(testlabels)

#  New code for including adaptive thresholding

unique_testlabels = list(set(testlabels))
sortlist = []

if len(testlabels) >2:
    # Normalize the results
    print "Normalized"
    if(max(results.values()) - min(results.values()))>0:
	 for i in unique_testlabels:
            norm_val = float((testlabels.count(i)- min(results.values())))/float((max(results.values()) - min(results.values())))
            if(norm_val > 0.3):
                sortlist.append(i)
    else:
        sortlist.append(i)
else:
    # Not normalized one
    # Put all the results
    for i in unique_testlabels:
        sortlist.append(i)
print "*****************************"
print "results"
print "*****************************"
print sortlist    
print results
sys.exit()
    






#print c rpus_tfidf
#print topic_patterns_list
#print llda.n_z_t[30]

#print "Result"
#print labelset[30]
#numpy.argsort(-llda.n_z_t[35,g])[:50]
#for g in numpy.argsort(-llda.n_z_t[30])[:50]:
#    print "%s: %.4f" % (llda.vocas[g], llda.n_z_t[30,g]) 

n_m_z_indices = np.nonzero(llda.n_m_z)
n_z_t_indices = np.nonzero(llda.n_z_t)
denom_b = llda.n_z_t.sum(axis=1) + len(llda.vocas) * llda.beta

# commented by praveen
# created the list of tuples for (topic, word, frequency)
n_t_w = []
n_z_t_indices_0 = n_z_t_indices[0]
n_z_t_indices_1 = n_z_t_indices[1]
for i in range(len(n_z_t_indices[0])):
    n_t_w.append([n_z_t_indices_0[i],n_z_t_indices_1[i], llda.n_z_t[n_z_t_indices_0[i],n_z_t_indices_1[i]]])

print "Topics"
#for k in range(len(llda.n_z_t[0]))

print n_t_w
print len(n_t_w)
sys.exit()

n_z_indices = np.nonzero(llda.n_z)


phi = llda.phi()

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for corp in corpus:
    for token in corp:
        frequency[token] += 1

texts = [[token for token in corp if frequency[token] > 1]
         for corp in corpus]

dictionary = corpora.Dictionary(texts)

corpus = [dictionary.doc2bow(text) for text in texts]

#tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model
#num_topics = len(labelset_deli)
#corpus_tfidf = tfidf[corpus]

lsi = models.LsiModel(corpus, id2word=dictionary, num_topics = len(labelset_deli)) # initialize an LSI transformation

print "Test labelling"
print corpus[10]
print labels[10]

#print llda.complement_label(labels[10])

test_doc = "How to do Programming in C"

#test_doc = corpus[10]

vec_bow = dictionary.doc2bow(test_doc.lower().split())
vec_lsi = lsi[vec_bow] # convert the query to LSI space
print(vec_lsi)

index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and index it

sims = index[vec_lsi] # perform a similarity query against the corpus
#print(list(enumerate(sims)))

sims = sorted(enumerate(sims), key=lambda item: -item[1])

print labels_r[sims[0][0]]
print labels_r[sims[1][0]]
sys.exit()

testdoc = [llda.term_to_id(term) for term in test_doc]
testlabel = llda.complement_label(None)

test_n_z = numpy.zeros(len(labelset) , dtype=int)
test_z_n = [numpy.random.multinomial(1, testlabel / testlabel.sum()).argmax() for x in range(len(testdoc))]

for i in range(options.iteration):
    testlabels = llda.inference_query(testdoc)
print testlabels

res =  sorted(range(len(testlabels)), key=lambda i: testlabels[i])[-3:]

print res
print labelset[res[0]]
print labelset[res[1]]
print labelset[res[2]]

#test_labels = llda.inference_query(['indonesia', 'rejected', 'world', 'bank', 'recommendations', 'for', 'sweeping', 'reforms', 'to', 'its', 'farm', 'economy', 'as', 'the', 'country'])

#print phi
#sys.exit()
#for k, label in enumerate(labelset):
#    print "\n-- label %d : %s" % (k, label)
