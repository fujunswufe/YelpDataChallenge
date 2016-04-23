__author__ = 'fujun'

from nltk import sent_tokenize
import json
import io

if __name__ == "__main__":
    review_file = io.open("/Users/fujun/GitHub/YelpDataChallenge/data/tip_review.json", 'r', encoding='utf8')

    index = 1

    for line in review_file:
        json_decode = json.loads(line.strip("\n"))

        review = json_decode.get("review")
        # print "type:" + str(type(review))
        tip = json_decode.get("tip")

        sentences = sent_tokenize(review)
        # print str(sentences[0])
        first = sentences[0]

        system = "system/task" + str(index) + "_englishSyssum" + str(index) + ".txt"
        reference = "reference/task" + str(index) + "_englishReference" + str(index) + ".txt"

        system_output = io.open(system, "w", encoding='utf8')
        reference_output = io.open(reference, "w", encoding='utf8')

        system_output.write(unicode(first))
        reference_output.write(unicode(tip))

        system_output.close()
        reference_output.close()

        index += 1

    review_file.close()


