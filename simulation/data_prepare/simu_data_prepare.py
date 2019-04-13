import json
import os
import random

from sklearn.model_selection import train_test_split

prefix = '/home/next/cr_repo/bf/conv_data/split/'

name_list = ['ddl', 'ddl2', 'ddl3', 'k', 'zh', 'k2']

file_name = 'data_labeled_{}.dat'

data_list = []
user_list = []
movie_list = []
for name in name_list:
    file_path = prefix + file_name.format(name)
    i = 1
    with open(file_path, 'r') as w:
        for line in w:
            line = line.strip()
            line = line.replace(',', ' ')
            line = line.replace('.', '')
            line = line.replace('。', '')
            line = line.replace('?', '')
            line = line.replace('\'s', '')
            line = line.replace('`', '')
            line = line.replace('\'', '')

            if i % 7 == 0:
                data_list.append(line)
            elif i % 8 == 0:
                data_list.append(line)
            elif i % 9 == 0:
                data_list.append(line)
                data_list.append(line.replace('audience', 'critic'))
            elif i % 10 == 0:
                data_list.append(line.replace('<genres>', '<genre>'))
            i = i % 11
            i = i + 1


def chunks(arr, n):
    return [arr[i:i+n] for i in range(0, len(arr), n)]


j = 1
data_1genre = []
data_2genre = []
data_3genre = []
data_4genre = []
data_5genre = []

data_5sentences = chunks(data_list, 5)

for data_list in data_5sentences:
    genre_sentence = data_list[4]
    genre_num = genre_sentence.count('<genre>')
    if genre_num == 1:
        data_1genre.append(data_list)
    elif genre_num == 2:
        data_2genre.append(data_list)
    elif genre_num == 3:
        data_3genre.append(data_list)
    elif genre_num == 4:
        data_4genre.append(data_list)
    elif genre_num == 5:
        data_5genre.append(data_list)
    else:
        print('error:', data_list)

data_genres = [data_1genre, data_2genre, data_3genre, data_4genre, data_5genre]

data_list = []
with open(os.path.expanduser('~/path/mv/movie_rating'), 'r') as f:
    for line in f:
        line = json.loads(line)
        data_list.append(line)


# get part of data list
data_list, useless, a, b = train_test_split(data_list, [0] * len(data_list), test_size=0.96, random_state=1)


data_final = []
for data in data_list:
    director = str(data['director'])
    genres = data['genres'].split('|')
    critic_rating = str(data['critic_rating'])
    country = str(data['country'])
    audience_rating = str(data['audience_rating'])
    movie = str(data['movie'])
    user = str(data['user'])

    if len(genres) > 5:
        genres = genres[:5]
    try:
        template = random.sample(data_genres[len(genres)-1], 1)[0]
    except Exception as e:
        print(e)
        print(data)
        print(genres)

    data_json_list = []

    value = 'co_' + country
    country_sentence = template[0]
    country_sentence = country_sentence.replace('<country>', value)
    sentence_split = country_sentence.split()
    tags = ['O'] * len(sentence_split)
    try:
        tag_start_index = sentence_split.index(value)
    except Exception as e:
        print(sentence_split)
        print(value)
    tags[tag_start_index] = 'B-' + 'country'
    data_json_list.append({'key': country_sentence, 'value': value, 'tags': tags, 'user': user, 'movie': movie})

    value = 'di_' + director
    director_sentence = template[1]
    director_sentence = director_sentence.replace('<director>', value)
    sentence_split = director_sentence.split()
    tags = ['O'] * len(sentence_split)
    try:
        tag_start_index = sentence_split.index(value)
    except Exception as e:
        print(sentence_split)
        print(value)

    tags[tag_start_index] = 'B-' + 'director'

    data_json_list.append({'key': director_sentence, 'value': value, 'tags': tags,  'user': user, 'movie': movie})

    value = 'au_' + audience_rating
    audience_sentence = template[2]
    audience_sentence = audience_sentence.replace('<audience>', value)
    sentence_split = audience_sentence.split()
    tags = ['O'] * len(sentence_split)
    try:
        tag_start_index = sentence_split.index(value)
    except Exception as e:
        print(sentence_split)
        print(value)

    tags[tag_start_index] = 'B-' + 'audience_rating'

    data_json_list.append({'key': audience_sentence, 'value': value, 'tags': tags,  'user': user, 'movie': movie})

    value = 'cr_' + critic_rating
    critic_sentence = template[3]
    critic_sentence = critic_sentence.replace('<critic>', value)
    sentence_split = critic_sentence.split()
    tags = ['O'] * len(sentence_split)
    try:
        tag_start_index = sentence_split.index(value)
    except Exception as e:
        print(sentence_split)
        print(value)

    tags[tag_start_index] = 'B-' + 'critic_rating'

    data_json_list.append({'key': critic_sentence, 'value': value, 'tags': tags,  'user': user, 'movie': movie})

    genre_sentece = template[4]
    tags = ['O'] * len(genre_sentece.split())
    value_list = []
    for genre in genres:
        value = 'ge_' + genre
        value_list.append(value)
        genre_sentece = genre_sentece.replace('<genre>', value, 1)
        sentence_split = genre_sentece.split()
        try:
            tag_start_index = sentence_split.index(value)
        except Exception as e:
            print(sentence_split)
            print(value)
        tags[tag_start_index] = 'B-' + 'genre'

    data_json_list.append({'key': genre_sentece, 'value': value_list, 'tags': tags,  'user': user, 'movie': movie})

    data_final.append(data_json_list)


with open('/home/next/cr_repo/bf/conv_data/test2.dat', 'w') as q:
    for five_sentences in data_final:
        for sentence in five_sentences:
            q.write(json.dumps(sentence) +'\n')
        # q.write('\n')



