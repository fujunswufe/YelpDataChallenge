__author__ = 'fujun'

import config
import json
import io
from nltk import sent_tokenize
from cStringIO import StringIO


def top(biz_path, review_path, tip_path, k=5000):

    biz_set = set()  # hash set for more than 10,000 business id

    biz_input = io.open(biz_path, 'r', encoding='utf8')
    for line in biz_input:
        biz_set.add(line.strip())
    biz_input.close()

    review_dict = {}  # key the biz id in biz_set, value is the list of review sentences
    for biz_id in biz_set:
        # review_dict[biz_id] = []
        # review_dict[biz_id] = StringIO()
        review_dict[biz_id] = []

    review_input = open(review_path, 'r')
    for line in review_input:
        json_decode = json.loads(line.strip())

        biz_id = json_decode.get("business_id")  # only store the business id in biz_set

        if biz_id in biz_set:
            # review_dict[biz_id].extend(sent_tokenize(json_decode.get("text")))
            # review_dict[biz_id].write(json_decode.get("text") + " ")
            review_dict[biz_id].append(json_decode.get("text"))
    review_input.close()

    sort_dict = {}  # key is the length of review sentences, and value is a list of business id
    for key in review_dict:
        # l = len(review_dict[key])
        # l = len(sent_tokenize(review_dict[key].getvalue()))
        l = 0
        for sents in review_dict[key]:
            l += len(sent_tokenize(sents))
        if l in sort_dict:
            sort_dict[l].append(key)
        else:
            sort_dict[l] = [key]

    biz_list = []  # use a list to store all biz_id in descending order (# of sentences)
    for key in sorted(sort_dict.keys(), reverse=True):
        biz_list.extend(sort_dict[key])

    biz_set_k = set()  # biz_set_k is the top k business id
    for i in range(0, k):
        biz_set_k.add(biz_list[i])

    tip_dict = {}  # key is in biz_set_k and value is list of tips
    for biz_id in biz_set_k:
        # tip_dict[biz_id] = StringIO()
        tip_dict[biz_id] = []

    tip_input = io.open(tip_path, 'r', encoding='utf8')
    for line in tip_input:
        json_decode = json.loads(line.strip())

        biz_id = json_decode.get("business_id")

        if biz_id in biz_set_k:
            # tip_dict[biz_id].append(json_decode.get("text"))
            # tip_dict[biz_id].write(json_decode.get("text") + " ")
            tip_dict[biz_id].append(json_decode.get("text"))

    output_file = open("full_review_tip.json", "w")

    # tip_dict, review_dict
    for biz_id in biz_set_k:
        temp_dict = {"business_id": biz_id, "review": " ".join(review_dict[biz_id]), "tip": " ".join(tip_dict[biz_id]), "user_id": biz_id}
        output_file.write(json.dumps(temp_dict) + "\n")
    output_file.close()

if __name__ == '__main__':
    # business = config.BIZ_ID_PATH
    # review = config.FULL_YELP_REVIEW_PATH
    # tip = config.FULL_YELP_TIP_PATH

    top("../../data/businessID.txt", '../../data/yelp/yelp_academic_dataset_review.json', '../../data/yelp/yelp_academic_dataset_tip.json', 5000)