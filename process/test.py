__author__ = 'fujun'

import json
import re
from string import maketrans
from string import punctuation
from nltk import word_tokenize

regex = re.compile('[%s]' % re.escape(punctuation))


def comp_len(tip):
    # regex.sub('', tip)
    return len(word_tokenize(regex.sub('', tip).strip()))

if __name__ == "__main__":
    s = 'where, apple... !!! time... *()  +_ %$ #'
    print comp_len(s)