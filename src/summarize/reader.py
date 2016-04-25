# encoding: utf-8
import os
import io
from gensim import corpora, models, similarities
import re
from summarize.config import *
from nltk import sent_tokenize
from nltk import word_tokenize

__author__ = 'Memray'

import json


stopwords = []

def load_stopword(path):
    stop_word = []
    try:
        stopword_file = open(path, 'r')
        stop_word = [line.strip() for line in stopword_file]
    except:
        print('Error occurs when loading STOPWORD')
    return stop_word

class Document:
    def __init__(self, review, tip, user_id, business_id):
        self.tip = tip
        self.review = review
        self.business_id = business_id
        self.user_id = user_id

    def to_json(self):
        return json.dumps(self.__dict__)

    @classmethod
    def from_json(cls, json_str):
        json_dict = json.loads(json_str)
        return cls(**json_dict)

def generate_dict_corpus_all_review():
    '''
    generate the gensim dict&corpus on the whole review corpus
    :return:
    '''

    print('Generating new dict and corpus on all Yelp reviews')

    review_file = open(FULL_YELP_REVIEW_PATH, 'r')
    # output_review = open("review.json", 'w')
    # output_tip = open("tip.json", 'w')

    texts = []
    stoplist = load_stopword(STOPWORD_PATH)

    count = 0

    for line in review_file:
        count += 1
        if count % 10000 ==0:
            print(count)
        json_review = json.loads(line.strip())

        text = json_review.get("text").decode('utf-8').lower()
        # tokenize and clean. Split non-word&number: re.sub(r'\W+|\d+', '', word.decode('utf-8')). Keep all words:r'\d+'
        tokens = [re.sub(r'\W+|\d+', '', word) for word in text.split()]
        # remove stop words and short tokens
        tokens = [token for token in tokens if ((not token.strip()=='') and (not token in stoplist))]
        # stemming, experiment shows that stemming works nothing...
        # if (stemming):
        #     stemmer = PorterStemmer()
        #     texts = [[ stemmer.stem(token) for token in text] for text in texts]
        texts.append(tokens)

    review_file.close()

    # remove words that appear only once
    # from collections import defaultdict
    # frequency = defaultdict(int)
    # for token in tokens:
    #     frequency[token] += 1
    # for text in texts:
    #     tokens = []
    #     for token in text:
    #         if (frequency[token] > 1):
    #             tokens.append(token)
    #     text = tokens
    # texts = [[token for token in text if (frequency[token] > 1)] for text in texts]

    print('Corpus preprocessing and counting complished!')

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=5)

    dictionary.save(DICT_PATH) # store the dictionary, for future reference
    dictionary.save_as_text(DICT_TXT_PATH)
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize(CORPUS_PATH, corpus) # store to disk, for later use
    print('Generating dict and corpus complished!')


def load_dict_corpus_all_review():
    '''
    return the gensim dict&corpus on the whole review corpus
    :return: dict&corpus
    '''
    if not (os.path.isfile(DICT_PATH) and os.path.isfile(CORPUS_PATH)):
        generate_dict_corpus_all_review()
    print('Reading dict & corpus')
    dict = corpora.Dictionary.load(DICT_PATH)
    corpus = corpora.MmCorpus(CORPUS_PATH)
    print('Reading complicated')
    return corpus, dict


def generate_summary_data():
    document_list = []

    '''
    load tips
    '''
    input_file = open(FULL_YELP_TIP_PATH, 'r')
    tip_dict = {}
    for line in input_file:
        json_tip = json.loads(line.strip())
        tup = (json_tip.get("user_id"), json_tip.get("business_id"))
        if tup not in tip_dict:
            tip_dict[tup] = json_tip.get("text").strip().lower()
        else:
            tip_dict[tup] = tip_dict[tup] + json_tip.get("text").strip().lower()

    input_file.close()

    print("length: ", len(tip_dict))

    # tip_file = open("tip_file.txt", "w")
    # for k, v in my_dict.items():
    #     if len(v) >= 2:
    #         tip_file.write(str(k) + str(v) + "\n")
    # tip_file.close()


    '''
    load reviews
    '''
    review_file = open(FULL_YELP_REVIEW_PATH, 'r', encoding='utf-8')
    # output_review = open("review.json", 'w')
    # output_tip = open("tip.json", 'w')

    count_with_tip = 0
    count_found_tip = 0
    output_string = ''

    for line in review_file:
        json_review = json.loads(line.strip())
        tup = (json_review.get("user_id"), json_review.get("business_id"))
        text = json_review.get("text").lower()

        # check whether this review contains a tip, ignore if not
        if tup in tip_dict:
            count_with_tip += 1
            tip = tip_dict[tup]
            doc = Document(text, tip, json_review.get("user_id"), json_review.get("business_id"))
            document_list.append(doc)
            if (text.find(tip) is not -1) and (not text == tip) :
                count_found_tip += 1
                output_string += doc.to_json()+'\n'
    review_file.close()
    # output_review.close()
    # output_tip.close()
    print(count_with_tip)
    print(count_found_tip)

    output_file = open(DATA_PATH, 'w')
    output_file.write(output_string.decode('utf-8'))
    output_file.close()


def load_yelp_training_data():
    '''
    Read the data and return the training data(summaries which contain a tip)
    If file does not exist, generate and save to file in a Json format
    :return:
    '''
    if not os.path.isfile(DATA_PATH):
        print('Start to generate summary&tip data')
        generate_summary_data()

    print('Start to read summary&tip data')
    doc_list = []
    data_file = open(DATA_PATH, 'r')
    for line in data_file:
        doc = Document.from_json(line)
        doc.tip = doc.tip.replace('\n','')
        doc.review = doc.review.replace('\n','')
        if len(doc.tip.strip())>20:
            doc_list.append(doc)

    return doc_list


def load_lda(corpus, dictionary):
    '''
    Load lda from file, or create/train a new one
    :return:
    '''
    if not os.path.isfile(LDA_MODEL_PATH):
        # Online LDA: extract 100 LDA topics, using 1 pass and updating once every 1 chunk (10,000 documents)
        print('Creating LDA..')
        lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=50, update_every=1, chunksize=10000, passes=1)
            # extract 100 LDA topics, using 20 full passes, no online updates
        #lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=0, passes=20)
        lda.save(LDA_MODEL_PATH)
        print('LDA finished..')

    print('Loading LDA model')
    lda = models.LdaModel.load(LDA_MODEL_PATH)
    return lda


def load_tfidf(corpus, dictionary):
    if not os.path.isfile(TFIDF_MODEL_PATH):
        print('Creating TF-IDF')
        tfidf = models.TfidfModel(corpus)
        print('TF-IDF created')
        tfidf.save(TFIDF_MODEL_PATH)

    print('Loading TF-IDF model')
    tfidf = models.TfidfModel.load(TFIDF_MODEL_PATH)
    return tfidf
# doc_list = get_data()
# print(len(doc_list))

def get_review_sentences():
    '''
    Read the yelp review and return after sentence segmentattion
    :return:
    '''
    review_file = io.open(FULL_YELP_REVIEW_PATH, 'r', encoding='utf-8')
    count_sentence = 0
    sentences = []

    for line in review_file:
        json_review = json.loads(line.strip())
        text = json_review.get("text").replace('\n','').lower()

        raw_sentences = sent_tokenize(text)
        for raw_sentence in raw_sentences:
            if len(raw_sentence.strip()) > 0:
                sent_tokens = word_tokenize(raw_sentence)
                sentences.append(sent_tokens)
    return sentences

def load_w2v(corpus, dictionary):
    '''
    Return the trained Word2Vec model
    Train a model if model doesn't exist yet
    :param corpus:
    :param dictionary:
    :return:
    '''
    if not os.path.isfile(W2V_MODEL_PATH):
        num_features = 300    # Word vector dimensionality
        min_word_count = 5    # Minimum word count
        num_workers = 5       # Number of threads to run in parallel
        window = 5          # Context window size
        downsampling = 1e-5   # Downsample setting for frequent words
        print("Training the word2vec model!")
        sents = get_review_sentences()
        # Initialize and train the model (this will take some time)
        model = models.Word2Vec(sents, workers=num_workers, \
                    size=num_features, min_count = min_word_count, \
                    window = window, sample = downsampling)

        # If you don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        model.init_sims(replace=True)

        # It can be helpful to create a meaningful model name and
        # save the model for later use. You can load it later using Word2Vec.load()
        model.save(W2V_MODEL_PATH)
        tfidf = models.Word2Vec(corpus)
        print('Word2vec model created!')

    print('Loading word2vec model')
    w2v = models.Word2Vec.load(W2V_MODEL_PATH)
    print('Loading word2vec model complished!')
    return w2v

def export_to_attention_model_data_format():
    doc_list = load_yelp_training_data()
    print(len(doc_list))
    # summary_file = io.open(ATTENTION_MODEL_SUMMARY_PATH, 'w', encoding='utf-8')
    # document_file = io.open(ATTENTION_MODEL_DOCUMENT_PATH, 'w', encoding='utf-8')
    document_summary_file = io.open(ATTENTION_MODEL_DOCUMENT_SUMMARY_PATH, 'w', encoding='utf-8')
    for doc in doc_list:
        # summary_file.write(doc.tip+'\n')
        # document_file.write(doc.review+'\n')
        document_summary_file.write(doc.tip+'\t'+doc.review+'\n')
    # summary_file.close()
    # document_file.close()
    document_summary_file.close()

export_to_attention_model_data_format()
