# encoding: utf-8
import numpy

__author__ = 'Memray'

from summarize.reader import *
from summarize.config import *
from gensim import corpora, models, similarities
import re
from nltk import sent_tokenize
from nltk import word_tokenize

def get_sentence_tokens(text):
    '''
    Given a text(review), return the token list of each sentence
    :param text:
    :return:
    '''
    sentences = sent_tokenize(text)

    sent_tokens = []
    for sentence in sentences:
        sent_token = word_tokenize(sentence)
        sent_token = [token for token in sent_token if ((not token.strip()=='') and (not token in stopwords))]
        sent_tokens.append(sent_token)
    # remove stop words and short tokens

    # stemming, experiment shows that stemming works nothing...
    # if (stemming):
    #     stemmer = PorterStemmer()
    #     texts = [[ stemmer.stem(token) for token in text] for text in texts]
    return sent_tokens


def build_representation(doc_list, method = 'tfidf'):
    '''
    Load corpus and convert to Dictionary and Corpus of gensim
    :param doc_list:
    :return:
    '''
    corpus, dictionary = load_dict_corpus_all_review()
    print(dictionary)

    count = 0

    for doc in doc_list:
        doc.sent_tokens = get_sentence_tokens(doc.review)

    if method=='tfidf':
        tfidf = load_tfidf(corpus, dictionary)
        for doc in doc_list:
            doc.vector = []
            for sent_tokens in doc.sent_tokens:
                doc.vector.append(tfidf[dictionary.doc2bow(sent_tokens)])
                # count += 1
                # if(count % 1000==0):
                #     print(count)
    elif method=='lda':
        lda = load_lda(corpus, dictionary)
        for doc in doc_list:
            doc.vector = []
            for sent_tokens in doc.sent_tokens:
                doc.vector.append(lda[dictionary.doc2bow(sent_tokens)])

    elif method=='word2vec':
        w2v = load_w2v(corpus, dictionary)
        # count = 0
        for doc in doc_list:
            # count += 1
            # if count % 100 == 0:
            #     print(count)
            doc.vector = []
            # iterate each sentence
            for sent_tokens in doc.sent_tokens:
                sent_vec = numpy.zeros((300,))
                tokens = [x for x in sent_tokens if x in w2v.vocab]
                # iterate each token in sentence
                for token in tokens:
                    sent_vec += w2v[token]
                sent_vec = sent_vec/len(tokens)
                doc.vector.append(sent_vec)


if __name__=='__main__':
    document_list = load_yelp_training_data()
    # build_representation(document_list, 'tfidf')
    build_representation(document_list, 'lda')
    # build_representation(document_list, 'word2vec')
    print()