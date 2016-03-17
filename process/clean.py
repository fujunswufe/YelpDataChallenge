__author__ = 'fujun'
# -*- coding: utf-8 -*-

import json
from nltk import sent_tokenize
from nltk import word_tokenize

"""
these codes are mainly for cleaning the test. Doing sentence segmentation and clean raw text.
"""

if __name__ == "__main__":

    review_file = open("yelp_academic_dataset_review.json", 'r')
    review_clean = open("review_clean.txt", "w")

    for line in review_file:
        json_decode = json.loads(line.strip("\n"))
        text = json_decode.get("text")

        sentences = sent_tokenize(text)
        #  maybe there are some non-english words 
        for sent in sentences:
            sent_tokens = word_tokenize(sent)
            review_clean.write(sent_tokens + "\n")

    review_file.close()
    review_clean.close()