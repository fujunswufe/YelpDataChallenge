import json

if __name__ == "__main__":
    input_file = open('yelp_academic_dataset_tip.json', 'r')
    my_dict = {}
    for line in input_file:
        json_decode = json.loads(line.strip("\n"))
        tup = (json_decode.get("user_id"), json_decode.get("business_id"))
        if tup not in my_dict:
            my_dict[tup] = [json_decode.get("text")]
        else:
            my_dict[tup].append(json_decode.get("text"))

    input_file.close()

    print "length: ", len(my_dict)

    # tip_file = open("tip_file.txt", "w")
    # for k, v in my_dict.items():
    #     if len(v) >= 2:
    #         tip_file.write(str(k) + str(v) + "\n")
    # tip_file.close()

    review_file = open("yelp_academic_dataset_review.json", 'r')

    # output_review = open("review.json", 'w')
    # output_tip = open("tip.json", 'w')



    for line in review_file:
        json_decode = json.loads(line.strip("\n"))
        tup = (json_decode.get("user_id"), json_decode.get("business_id"))
        text = json_decode.get("text")

        if tup in my_dict:
            l = my_dict[tup]
            for items in l:
                # if text.get("text").find(items) is not -1:
                if text.find(items) is not -1:
                    pass
                    # output_review.write(line)
                    # temp_dict = {"user_id": tup[0], "business_id": tup[1], "text": items}
                    # output_tip.write(json.dumps(temp_dict) + "\n")

    review_file.close()
    # output_review.close()
    # output_tip.close()