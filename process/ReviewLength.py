__author__ = 'fujun'

import json
import re
from string import punctuation
from nltk import word_tokenize

regex = re.compile('[%s]' % re.escape(punctuation))


def comp_len(tip):
    return len(word_tokenize(regex.sub('', tip).strip()))

if __name__ == "__main__":

    count = 0.0
    review_len = 0.0
    review_file = open("yelp_academic_dataset_review.json", 'r')
    for line in review_file:
        json_decode = json.loads(line.strip("\n"))
        text = json_decode.get("text")
        count += 1.0
        review_len += comp_len(text)
    review_file.close()

    print "average length per review: ", str(review_len/count)
