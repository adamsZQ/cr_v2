
import json


def rating_2id(rating, max):
    if 5 * rating <= max:
        rating = 0
    elif 5 * rating <= 2 * max:
        rating = 1
    elif 5 * rating <= 3 * max:
        rating = 2
    elif 5 * rating <= 4 * max:
        rating = 3
    elif 5 * rating <= 5 * max:
        rating = 4

    return str(rating)


data_tolabel_list = []
with open('/home/next/cr_repo/movie_data_cleaned', 'r') as f:
    for line in f:
        data_tolabel = {}
        line = json.loads(line)
        # line.pop('id')
        data_tolabel = line

        critic_rating = line['critic_rating']
        audience_rating = line['audience_rating']

        critic_rating = rating_2id(float(critic_rating), 10.0)
        audience_rating = rating_2id(float(audience_rating), 5.0)

        data_tolabel['critic_rating'] = critic_rating
        data_tolabel['audience_rating'] = audience_rating

        data_tolabel_list.append(data_tolabel)

with open('/home/next/cr_repo/movie_data_cleaned_normal', 'w') as w:
    for line in data_tolabel_list:
        line = json.dumps(line)
        w.write(line)
        w.write('\n')