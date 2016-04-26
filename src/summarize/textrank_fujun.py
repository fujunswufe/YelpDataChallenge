__author__ = 'fujun'

import nltk
import itertools
import networkx as nx
from operator import itemgetter
from math import log
import string
from config import *
from representation import *


stopwords = set()  # store all the stopwords in a hash set
stop_word_input = open(STOPWORD_PATH, 'r')
for word in stop_word_input:
    stopwords.add(word.strip())
stop_word_input.close()

table = string.maketrans("", "")  # used for remove punctuation, reduce some memory


# process the raw text of a node
def process(node):

    tokens = nltk.word_tokenize(node.lower().translate(table, string.punctuation))

    for token in tokens:
        if token in stopwords:
            tokens.remove(token)

    return tokens


# type could be "word2vec", "tfidf", "lda"
def graph_weight(node1, node2, rep="rawText"):

    if rep == "rawText":  # node1 and node2 are just raw text

        result = len(set(first).intersection(second)) / float(log(len(first)) + log(len(second)))

        return result

    elif rep == "word2vec":  # vectors, cosine similarity
        pass

    elif rep == "tfidf":  # vectors, other measures
        pass

    elif rep == "lda":  # vectors, cosine similarity
        pass

    else:
        assert "wrong parameter for rep"


# def lDistance(firstString, secondString):
#     """
#     Function to find the Levenshtein distance between two words/sentences -
#     gotten from http://rosettacode.org/wiki/Levenshtein_distance#Python"
#     """
#     if len(firstString) > len(secondString):
#         firstString, secondString = secondString, firstString
#     distances = range(len(firstString) + 1)
#     for index2, char2 in enumerate(secondString):
#         newDistances = [index2 + 1]
#         for index1, char1 in enumerate(firstString):
#             if char1 == char2:
#                 newDistances.append(distances[index1])
#             else:
#                 newDistances.append(1 + min((distances[index1], distances[index1+1], newDistances[-1])))
#         distances = newDistances
#     return distances[-1]


# nodes should be list of raw sentences
def build_graph_raw(nodes):

    """
    nodes - actually the sentences in our sentence extraction
    nodes - list of hashable that represents the nodes of the graph
            hashable: string, numbers, tuples
    """

    g = nx.Graph()  # initialize an undirected graph

    g.add_nodes_from(nodes)

    node_pairs = list(itertools.combinations(nodes, 2))

    # add edges to the graph (weighted by normalizing similarity)
    for pair in node_pairs:

        node1 = pair[0]
        node2 = pair[1]

        first = process(node1)
        second = process(node2)

        if len(first) >= 2 and len(second) >= 2:
            w = graph_weight(first, second)
            if w > 0:
                gr.add_edge(first, second, weight = w)

    return G


# document should be a list of sentences
def extraction(document):
    # sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    # sentenceTokens = sent_detector.tokenize(text.strip())
    # sentenceTokens = document.sentences()
    # graph = buildGraph(sentenceTokens)

    graph = build_graph_raw(document)  #

    calculated_page_rank = nx.pagerank(graph, weight='weight')

    #most important sentences in ascending order of importance
    sentences = sorted(calculated_page_rank, key=calculated_page_rank.get, reverse=True)

    #return a 100 word summary
    summary = ' '.join(sentences)
    summaryWords = summary.split()
    summaryWords = summaryWords[0:101]
    summary = ' '.join(summaryWords)

    return summary


