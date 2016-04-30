# encoding: utf-8
import numpy

__author__ = 'Memray'

from reader import *
from config import *
from extraction import *
from gensim import corpora, models, similarities
import re
from nltk import sent_tokenize
from nltk import word_tokenize
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix, vstack

def conver_to_vector(array,N):
    sent_vec = numpy.zeros(shape=(1,N))
    for tup in array:
        sent_vec[0,tup[0]]=tup[1]
    return sent_vec

def conver_to_sparsevector(array,N):
    sent_vec = csr_matrix((1,N))
    for tup in array:
        sent_vec[0,tup[0]]=tup[1]
    return sent_vec


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

    print('Tokenizing, may take a while')
    for doc in doc_list:
        doc.sent_tokens = get_sentence_tokens(doc.review)
    print('Tokenizing accomplished')

    if method=='tfidf':
        tfidf = load_tfidf(corpus, dictionary)
        number =1
        N = 100000
        for doc in doc_list:
            doc.vector = numpy.zeros(shape=(1,N))
            for sent_tokens in doc.sent_tokens:
                sent_vec = conver_to_vector(tfidf[dictionary.doc2bow(sent_tokens)],N)
                doc.vector = numpy.r_[doc.vector, sent_vec]
            # doc.vector = numpy.delete(doc.vector, 0, 0)
            # index = extract_summary_nopenaty(doc.vector, 3, 1)
            # sentences = sent_tokenize(doc.review)
            # write_system_tip_path = "../../system/task"+str(number)+"_englishSyssum"+str(number)+".txt"
            # write_reference_tip_path ="../../reference/task"+str(number)+"_englishReference"+str(number)+".txt"
            # number = number+1
            # file_w_sys = open(write_system_tip_path , 'w')
            # file_w_ref = open(write_reference_tip_path , 'w')
            #
            # file_w_sys.write(sentences[index])
            # file_w_ref.write(doc.tip)
    elif method=='lda':
        N = 50
        lda = load_lda(corpus, dictionary)
        number = 1
        for doc in doc_list:
            doc.vector = numpy.zeros(shape=(1,N))

            for sent_tokens in doc.sent_tokens:
                sent_vec = conver_to_vector(lda[dictionary.doc2bow(sent_tokens)],N)
                doc.vector = numpy.r_[doc.vector, sent_vec]

            # doc.vector = numpy.delete(doc.vector, 0, 0)
            # index = extract_summary_nopenaty(doc.vector, 3, 1)
            # sentences = sent_tokenize(doc.review)
            # write_system_tip_path = "../../system/task"+str(number)+"_englishSyssum"+str(number)+".txt"
            # write_reference_tip_path ="../../reference/task"+str(number)+"_englishReference"+str(number)+".txt"
            # number = number+1
            # file_w_sys = open(write_system_tip_path , 'w')
            # file_w_ref = open(write_reference_tip_path , 'w')
            #
            # file_w_sys.write(sentences[index])
            # file_w_ref.write(doc.tip)


    elif method=='word2vec':
        w2v = load_w2v(corpus, dictionary)
        # count = 0

        number = 1

        for doc in doc_list:
            # count += 1
            # if count % 100 == 0:
            #     print(count)
            doc.vector = numpy.zeros((300,))
            # iterate each sentence

            for sent_tokens in doc.sent_tokens:
                sent_vec = numpy.zeros((300,))
                tokens = [x for x in sent_tokens if x in w2v.vocab]
                # iterate each token in sentence
                for token in tokens:
                    sent_vec += w2v[token]
                sent_vec = sent_vec/len(tokens)
                doc.vector = numpy.c_[doc.vector, sent_vec]

            # doc.vector = numpy.delete(doc.vector, 0, 1)
            # doc.vector = doc.vector.transpose()
            # index = extract_summary_nopenaty(doc.vector, 3, 1)
            # sentences = sent_tokenize(doc.review)
            # write_system_tip_path = "../../system/task"+str(number)+"_englishSyssum"+str(number)+".txt"
            # write_reference_tip_path ="../../reference/task"+str(number)+"_englishReference"+str(number)+".txt"
            #
            # file_w_sys = open(write_system_tip_path , 'w')
            # file_w_ref = open(write_reference_tip_path , 'w')
            #
            # file_w_sys.write(sentences[index])
            # file_w_ref.write(doc.tip)

            number= number+1


            # index = extract_summary_nopenaty(doc.vector, 3, 1)



if __name__=='__main__':
    document_list = load_yelp_training_data()
    # build_representation(document_list, 'tfidf')
    build_representation(document_list, 'tfidf')
    # build_representation(document_list, 'word2vec')

    print()