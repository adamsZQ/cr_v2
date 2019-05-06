import json
import os
import random

from sklearn.model_selection import train_test_split

from tools.simple_tools import chunks

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
                data_list.append(line.replace('<genre>', '<genres>'))
            i = i % 11
            i = i + 1

j = 1
data_1genre = []
data_2genre = []
data_3genre = []
data_4genre = []
data_5genre = []

data_5sentences = chunks(data_list, 5)

for data_list in data_5sentences:
    genre_sentence = data_list[4]
    genre_num = genre_sentence.count('<genres>')
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


with open('/home/next/cr_repo/entity_id/director_id.json', 'r') as f:
    director2id = json.load(f)
id2director = {value: key for key, value in director2id.items()}
with open('/home/next/cr_repo/entity_id/country_id.json', 'r') as f:
    country2id = json.load(f)
id2country = {value: key for key, value in country2id.items()}
with open('/home/next/cr_repo/entity_id/genre_id.json', 'r') as f:
    genre2id = json.load(f)
id2genre = {value: key for key, value in genre2id.items()}
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

    value = id2country[int(country)]
    value_length = len(value.split())
    country_sentence = template[0]
    tag_start_index = country_sentence.split().index('<country>')
    country_sentence = country_sentence.replace('<country>', value)
    sentence_split = country_sentence.split()
    tags = ['O'] * len(sentence_split)

    tags[tag_start_index] = 'B-' + 'country'
    if value_length != 1:
        for i in range(value_length - 1):
            tags[tag_start_index+i+1] = 'I-' + 'country'
    data_json_list.append({'key': country_sentence, 'value': value, 'tags': tags, 'user': user, 'movie': movie})

    value = id2director[int(director)]
    value_length = len(value.split())
    director_sentence = template[1]
    tag_start_index = director_sentence.split().index('<director>')
    director_sentence = director_sentence.replace('<director>', value)
    sentence_split = director_sentence.split()
    tags = ['O'] * len(sentence_split)

    tags[tag_start_index] = 'B-' + 'director'
    if value_length != 1:
        for i in range(value_length - 1):
            tags[tag_start_index+i+1] = 'I-' + 'director'

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
        value = id2genre[int(genre)]
        value_list.append(value)
        tag_start_index = genre_sentece.split().index('<genres>')
        genre_sentece = genre_sentece.replace('<genres>', value, 1)
        sentence_split = genre_sentece.split()
        tags[tag_start_index] = 'B-' + 'genres'

    data_json_list.append({'key': genre_sentece, 'value': value_list, 'tags': tags,  'user': user, 'movie': movie})

    data_final.append(data_json_list)


with open('/home/next/cr_repo/bf/conv_data/test_full_entity.dat', 'w') as q:
    for five_sentences in data_final:
        for sentence in five_sentences:
            q.write(json.dumps(sentence) +'\n')



