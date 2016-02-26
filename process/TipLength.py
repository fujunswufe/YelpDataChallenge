__author__ = 'fujun'

import json
import re
from string import maketrans
from string import punctuation
from nltk import word_tokenize
from string import translate

regex = re.compile('[%s]' % re.escape(punctuation))


def comp_len(yelp_tip):
    # ss = regex.sub('', yelp_tip).strip()
    # sl = word_tokenize(ss)
    # return len(sl)
    return len(word_tokenize(regex.sub('', yelp_tip).strip()))

# table = maketrans("", "")
#
#
# def comp_len(s):
#     # clean = translate(table, punctuation).strip()
#     return len(word_tokenize(s.translate(table, punctuation).strip()))

if __name__ == "__main__":
    input_file = open('yelp_academic_dataset_tip.json', 'r')
    tips_dict = {}
    for line in input_file:
        json_decode = json.loads(line.strip("\n"))
        tup = (json_decode.get("user_id"), json_decode.get("business_id"))
        if tup not in tips_dict:
            tips_dict[tup] = [json_decode.get("text")]
        else:
            tips_dict[tup].append(json_decode.get("text"))

    input_file.close()

    tip_review = {}
    review_file = open("yelp_academic_dataset_review.json", 'r')
    for line in review_file:
        json_decode = json.loads(line.strip("\n"))
        tup = (json_decode.get("user_id"), json_decode.get("business_id"))
        text = json_decode.get("text")

        if tup in tips_dict:
            tips = tips_dict[tup]
            for tip in tips:  # items
                if text.find(tip) is not -1:
                    if tup not in tip_review:
                        tip_review[tup] = [tip]
                    else:
                        tip_review[tup].append(tip)
    review_file.close()

    print "tips_dict: ", len(tips_dict)
    print "tip_review: ", len(tip_review)

    count = 0.0
    len_tip = 0.0

    for k, v in tips_dict.items():
        for tip in v:
            count += 1.0
            len_tip += comp_len(tip)

    print "count: ", str(count)
    print "length: ", str(len_tip)
    print "average length: ", str(len_tip/count)

    count_tip = 0.0
    count_non_tip = 0.0
    len_tip = 0.0
    len_non_tip = 0.0
    #
    for k, v in tips_dict.items():
        if k not in tip_review:  # tips not in reviews
            for tip in v:
                count_non_tip += 1
                len_non_tip += comp_len(tip)
        else:
            for tip in v:
                if tip in tip_review[k]:
                    count_tip += 1
                    len_tip += comp_len(tip)
                else:
                    count_non_tip += 1
                    len_non_tip += comp_len(tip)

    print "average length of tips in review: ", str(len_tip/count_tip)
    print "average length of tips not in review: ", str(len_non_tip/count_non_tip)