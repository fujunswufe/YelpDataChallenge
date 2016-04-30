__author__ = 'fujun'

import nltk
import itertools
import networkx
import io
import string
import json
import re

from nltk import sent_tokenize
from nltk import word_tokenize

from operator import itemgetter
from math import log
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy import sparse

from config import *
from representation import *
from reader import *

stopwords = set()  # store all the stopwords in a hash set
stop_word_input = open(STOPWORD_PATH, 'r')
for word in stop_word_input:
    stopwords.add(word.strip())
stop_word_input.close()

regex = re.compile('[%s]' % re.escape(string.punctuation))

# process the raw text of a node
def process(node):

    tokens = nltk.word_tokenize(regex.sub('', node).lower())
    
    for token in tokens:

        if token in stopwords:
            tokens.remove(token)

    return tokens


# type could be "word2vec", "tfidf", "lda"
#def graph_weight(node1, node2, method="rawText"):

    #if method == "rawText":  # node1 and node2 are a list of tokens
        ## print "node1: " + str(len(node1))
        ## print "node2: " + str(len(node2))
        #numerator = len(set(node1).intersection(node2))
        #denomiator = float(log(len(node1)) + log(len(node2)))
        
        ## print "numerator: " + str(numerator)
        ## print "denomiator: " + str(denomiator)

        ## if numerator == 0:
            ## return 0
        ## else:

        #return numerator / denomiator

    #elif method == "word2vec":  # vectors, cosine similarity

        #similarity = np.dot(node1, node2)
        #l1 = np.sqrt(np.sum(node1**2))
        #l2 = np.sqrt(np.sum(node2**2))
        #return similarity / (l1*l2)

    #elif method == "tfidf":  # vectors, other measures
        #pass

    #elif method == "lda":  # vectors, cosine similarity
        #pass

    #else:
        #assert "wrong parameter for rep"


# nodes should be list of raw sentences
# method = "tfidf", "word2vec", "lda"
def build_graph(doc_list, method="rawText", extract_n=1):

    """
    nodes - actually the sentences in our sentence extraction
    nodes - list of hashable that represents the nodes of the graph
            hashable: string, numbers, tuples
    """
    corpus, dictionary = load_dict_corpus_all_review()
    
    
    count = 1
    for doc in doc_list:
        print "count: " + str(count)
        count+=1
        doc.sent_tokens = get_sentence_tokens(doc.review)
        if(count>10):
            break
        
      # initialize an undirected graph

    # index = 0  # sentence in sentences_tokenize

    if method == "rawText":

        for node in doc_list:
            v = process(node)
            if len(v) >= 5:
                g.add_node(node, value=process(node))

            index += 1

    elif method == "tfidf":  # add long vector as value

        tfidf = load_tfidf(corpus, dictionary)
        print "load tfidf completed"
        doc_index = 1

        for doc in doc_list:
            
            g = networkx.Graph()
            
            #if len(doc.sent_tokens) > 500:  # choose review sentences less than 500
                #continue
            
            print "process doc_index: " + str(doc_index)
            
            # doc.vector = numpy.zeros((200,))
            
            index = 0

            for sent_tokens in doc.sent_tokens:

                g.add_node(index)
                index += 1
                print "intermediate g.nodes(): " + str(g.nodes())

                doc.vector = numpy.zeros(shape=(1,100000))
    
                for sent_tokens in doc.sent_tokens:
                    sent_vec = conver_to_vector(tfidf[dictionary.doc2bow(sent_tokens)],100000)
                    doc.vector = numpy.r_[doc.vector, sent_vec]
                doc.vector = numpy.delete(doc.vector, 0, 0) 
                
                doc.vector = sparse.csr_matrix(doc.vector)

            cosine_matrix = cosine_similarity(doc.vector, doc.vector)

            add_weight(g, cosine_matrix)
            print "before g.nodes(): " + str(g.nodes())
            calculated_page_rank = networkx.pagerank(g, weight="weight")
            indexes = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=False)

            print_summary(indexes, doc, extract_n, doc_index)
            doc_index += 1
            g.clear()
            print "after g.nodes(): " + str(g.nodes())  
                
    elif method == "word2vec":  # add long vector as value

        w2v = load_w2v(corpus, dictionary)
        print "load_w2v completed"
        doc_index = 1
        
        for doc in doc_list:
            
            g = networkx.Graph()
            
            #if len(doc.sent_tokens) > 500:  # choose review sentences less than 500
                #continue
            
            print "process doc_index: " + str(doc_index)
            
            doc.vector = numpy.zeros((300,))
            
            index = 0

            for sent_tokens in doc.sent_tokens:

                g.add_node(index)
                index += 1
                print "intermediate g.nodes(): " + str(g.nodes())

                sent_vec = numpy.zeros((300,))
                tokens = [x for x in sent_tokens if x in w2v.vocab]
                # iterate each token in sentence
                for token in tokens:
                    sent_vec += w2v[token]
                sent_vec = sent_vec/len(tokens)
                doc.vector = numpy.c_[doc.vector, sent_vec]

            doc.vector = numpy.delete(doc.vector, 0, 1)
            doc.vector = doc.vector.transpose()

            cosine_matrix = cosine_similarity_self(doc.vector)

            add_weight(g, cosine_matrix)
            print "before g.nodes(): " + str(g.nodes())
            calculated_page_rank = networkx.pagerank(g, weight="weight")
            indexes = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=False)

            print_summary(indexes, doc, extract_n, doc_index)
            doc_index += 1
            g.clear()
            print "after g.nodes(): " + str(g.nodes())

    elif method == "lda":  # add long vector as value

        lda = load_lda(corpus, dictionary)
        
        print "load_lda completed"
        doc_index = 1
        
        for doc in doc_list:
            
            g = networkx.Graph()
            
            #if len(doc.sent_tokens) > 500:  # choose review sentences less than 500
                #continue
            
            print "process doc_index: " + str(doc_index)
            
            # doc.vector = numpy.zeros((200,))
            
            index = 0

            for sent_tokens in doc.sent_tokens:

                g.add_node(index)
                index += 1
                #print "intermediate g.nodes(): " + str(g.nodes())

                doc.vector = numpy.zeros(shape=(1,200))
    
                for sent_tokens in doc.sent_tokens:
                    sent_vec = conver_to_vector(lda[dictionary.doc2bow(sent_tokens)],200)
                    doc.vector = numpy.r_[doc.vector, sent_vec]
                doc.vector = numpy.delete(doc.vector, 0, 0)

            cosine_matrix = cosine_similarity_self(doc.vector)

            add_weight(g, cosine_matrix)
            #print "before g.nodes(): " + str(g.nodes())
            calculated_page_rank = networkx.pagerank(g, weight="weight")
            indexes = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=False)

            print_summary(indexes, doc, extract_n, doc_index)
            doc_index += 1
            g.clear()
            #print "after g.nodes(): " + str(g.nodes())        
 
    else:
        assert "wrong parameter for method"

    # return g
    
def conver_to_vector(array,N):
    sent_vec = numpy.zeros(shape=(1,N))
    for tup in array:
        sent_vec[0,tup[0]]=tup[1]
    return sent_vec

def print_summary(indexes, doc, extract_n, doc_index):

    if len(indexes) < extract_n:
        extract_n = len(indexes)

    reference = "reference/task" + str(doc_index) + "_englishReference" + str(doc_index) + ".txt"
    reference_output = io.open(reference, "w", encoding='utf8')
    tips = sent_tokenize(doc.tip)

    for tip in tips:
        reference_output.write(tip + "\n")
    reference_output.close()

    sentences = sent_tokenize(doc.review)
    
    #print ""
    ## print "sentences length: " + str(len(sentences))
    #print ""
    #print "indexes: " + str(indexes)
    #print ""
    
    system = "system/task" + str(doc_index) + "_englishSyssum" + str(doc_index) + ".txt"
    system_output = io.open(system, "w", encoding='utf8')    
    for i in range(0, extract_n):
        #print "index: " + str(indexes[i])
        system_output.write(sentences[indexes[i]] + "\n")

    system_output.close()


def add_weight(g, cosine_matrix):
    
    n = len(cosine_matrix)
    
    print ""
    print "dimension: " + str(len(cosine_matrix))
    print ""    
    
    for i in range(0, n):
        for j in range(i, n):
            if i is not j:
                if not np.isnan(cosine_matrix[i][j]): 
                    g.add_edge(i, j, weight=cosine_matrix[i][j])
                    #print "i, j: " + str((i, j)) + " ,weight: " + str(cosine_matrix[i][j]) + " , first"
                else:
                    g.add_edge(i, j, weight=0)
                    #print "i, j: " + str((i, j)) + " ,weight: " + str(0) + ", second"

    #node_pairs = list(itertools.combinations(g.nodes(), 2))

    #for pair in node_pairs:
        #node1 = pair[0]
        #node2 = pair[1]

        #n1 = g.node[node1]
        #n2 = g.node[node2]

        ## if n1 >= 2 and n2 >= 2:
        ##     w = graph_weight(g.node[node1]['value'], g.node[node2]['value'], method="rawText")

        #g.add_edge(node1, node2, weight=cosine_matrix[n1][n2])


def cosine_similarity_self(A):
    similarity = np.dot(A, A.T)
    square_mag = np.diag(similarity)
    inv_square_mag = 1 / square_mag
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    inv_mag = np.sqrt(inv_square_mag)
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag
    return cosine

# document should be a list of sentences
# method = "word2vec", "lda", "tfidf"
# def extraction(document, method="rawText"):
#
#     # graph = build_graph(document, method)  # document is a list of sentences
#
#     calculated_page_rank = networkx.pagerank(graph, weight="weight")
#
#     # most important sentences in descending order of importance
#     sentences = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=False)
#
#     return sentences[0:4]


if __name__ == "__main__":

    document_list = load_yelp_training_data()   # review level, 17000
    build_graph(document_list, method="lda", extract_n=20)

    # system = "system/task" + str(index) + "_englishSyssum" + str(index) + ".txt"
    # reference = "reference/task" + str(index) + "_englishReference" + str(index) + ".txt"

    # business_input = io.open("full_review_tip.json", 'r', encoding='utf8')
    #
    # for i, line in enumerate(business_input):  # i is 0 based
    #
    #     if i < 4:
    #
    #         json_decode = json.loads(line.strip())
    #
    #         reviews = nltk.sent_tokenize(json_decode.get("review").encode('utf-8'))
    #
    #         # system = "system/task" + str(i+1) + "_englishSyssum" + str(i+1) + ".txt"
    #         # system_output = open(system, "w")
    #         #
    #         # reference = "reference/task" + str(i+1) + "_englishReference" + str(i+1) + ".txt"
    #         # reference_output = open(reference, "w")
    #
    #         system_summary = extraction(reviews)
    #
    #         # print system_summary
    #         print ""
    #         print "system summrary: "
    #         print ""
    #
    #         for sent in system_summary:
    #             print "sent: " + sent
    #
    #         # system_output.write(str(system_summary) + "\n")
    #         # system_output.close()
    #         #
    #         # reference_summary = sent_tokenize(json_decode.get("tip").decode('utf-8'))
    #         # for sent in reference_summary:
    #         #     reference_output.write(sent + "\n")
    #         # reference_output.close()
    #
    #     else:
    #         break