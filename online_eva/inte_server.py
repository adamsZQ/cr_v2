import argparse
import json
import os
import random
import sys
import time

from werkzeug.contrib.cache import SimpleCache
import numpy as np

import torch
from sklearn.model_selection import train_test_split
from torch import optim, nn

from belief_tracker.BiLSTM_CRF_nobatch import load_model
from belief_tracker.data.glove import Glove_Embeddings
from belief_tracker.data.training_data import get_simulate_data
from belief_tracker.train.bilstm_training import prepare_sequence
from recommend.knn_recommend.knn import KNN
from tools.data_transfer import DataTool
from tools.simple_tools import chunks, zip_data
from tools.sql_tool import select_by_attributes, select_genres, select_all_movie_genres, select_all
from flask import Flask

FILE_PREFIX = None
model_type = None
boundary_tags = None

if FILE_PREFIX is None:
    FILE_PREFIX = '~/cr_repo/'
if model_type is None:
    model_type = 'test_entity_data'
if boundary_tags is None:
    boundary_tags = False


HIDDEN_DIM = 20
bf_prefix = 'bf/'

FILE_PREFIX = os.path.expanduser(FILE_PREFIX)

data_path = bf_prefix + model_type + '/training_data'
sentences_data, tag_data, user_list, movie_list = (get_simulate_data(FILE_PREFIX, data_path))

# get word embeddings
glove_embeddings = Glove_Embeddings(FILE_PREFIX, data_path)
glove_embeddings.words_expansion()
word_embeddings = glove_embeddings.task_embeddings
word2id = glove_embeddings.task_word2id
tag2id = glove_embeddings.task_tag2id
id2word = glove_embeddings.task_id2word
id2tag = glove_embeddings.task_id2tag

# sentence data -> index
sentences_prepared = prepare_sequence(sentences_data, word2id, boundary_tags)
tag_prepared = prepare_sequence(tag_data, tag2id, boundary_tags)

# zip sentence, user and movie
data_zipped = zip_data(sentences_prepared, user_list, movie_list)
tag_chunk = chunks(tag_prepared, 5)

# load bf model
model_path = '/home/next/cr_repo/bf/test_entity_data/bilstm_crf_0.0001.pkl'
bf_model = load_model(model_path)
# TODO load word embedding
embedding_path = '/home/next/cr_repo/bf/test_entity_data/embedding0.00011859582542694813_enforcement.pkl'
word_embeds_weight = torch.load(embedding_path)
word_embeds = nn.Embedding.from_pretrained(word_embeds_weight, freeze=True)

# get part of datalist
# X_train, X_test, y_train, y_test = train_test_split(data_zipped, tag_chunk, test_size=0.96, random_state=1)
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
# print(len(X_train))
# X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=2)


actions = ['director', 'genres', 'critic_rating', 'country', 'audience_rating', 'recommendation']
# question sequence in training data
question_sequence = ['country', 'director', 'audience_rating', 'critic_rating', 'genres']
movie_genres = select_all_movie_genres()

# print(movie_genres)

'''
    get all id-genres
'''
movie_id_list = []
for movie_genre in movie_genres:
    if movie_genre[0] not in movie_id_list:
        movie_id_list.append(movie_genre[0])

data_list = select_all()

id_genres_list = [None] * (max(movie_id_list) + 1)

for data in data_list:
    movie_id = data[0]
    genres = data[4].split('|')

    genres = [int(genre) for genre in genres]
    id_genres_list[movie_id] = genres


with open('/home/next/cr_repo/entity2id.dat', 'r') as f:
    entity2id = json.load(f)

id2entity = {value: key for key, value in entity2id.items()}

def get_genres(movie_id):
    return id_genres_list[movie_id]


with open('/home/next/cr_repo/entity_id/director_id.json', 'r') as f:
    director2id = json.load(f)
id2director = {value: key for key, value in director2id.items()}
with open('/home/next/cr_repo/entity_id/country_id.json', 'r') as f:
    country2id = json.load(f)
id2country = {value: key for key, value in country2id.items()}
with open('/home/next/cr_repo/entity_id/genre_id.json', 'r') as f:
    genre2id = json.load(f)
id2genre = {value: key for key, value in genre2id.items()}

# movie_rating_list = []
# question_sequence = ['country', 'director', 'audience_rating', 'critic_rating', 'genres']
data_sentence_list = []
with open('/home/next/cr_repo/movie_rating', 'r') as m:
    for line in m:
        data_json = json.loads(line)
        data_sentence = {}
        five_sentences = ['which country do you like?', 'which director do you like', 'what audience rating do you like?',
                          'what critic rating do you like>', 'which genres do you like?']
        # data_sentence['origin'] = data_json
        data_sentence['user'] = data_json['user']
        data_sentence['movie'] = data_json['movie']

        data_json['country'] = id2country[data_json['country']]
        data_json['director'] = id2director[data_json['director']]

        genres_array = data_json['genres'].split('|')
        genres_list = []
        genres_str = ''
        for genre in genres_array:
            genre_entity = id2genre[int(genre)]

            genres_str = genres_str + '|' + genre_entity

        data_json['genres'] = genres_str

        data_sentence['origin'] = data_json
        data_sentence['five_sentences'] = five_sentences

        data_sentence_list.append(data_sentence)

def get_bf_result(action, answer):
    entity_asked = actions[action]
    answer_index = prepare_sequence([answer.split()], word2id, False)

    sentence = torch.tensor(answer_index).squeeze(0).long().to(device)
    # print(sentence)
    predict = bf_model(word_embeds, sentence)
    tags_pred_list = predict[1]

    entity_id_str_list = []
    entity_id_list = []
    entity_tag_list = []

    previous_tag = ''
    entity_queue = []
    entities_list = []
    for word, tag in zip(sentence, tags_pred_list):
        tag_name = id2tag[tag]
        word_name = id2word[word.tolist()]

        if tag_name != 'O':
            start_tag = tag_name.split('-')[0]
            tag_entity = tag_name.split('-')[1]

            if entity_asked == tag_entity:
                if start_tag == 'B':
                    if len(entity_queue) != 0:
                        entities_list.append(entity_queue.copy())
                        entity_queue = []
                    entity_queue.append(word_name)
                    previous_tag = 'B'
                elif start_tag == 'I' and previous_tag != 'O':
                    entity_queue.append(word_name)
                    previous_tag = 'I'
        else:
            previous_tag = 'O'
    entities_list.append(entity_queue.copy())
    for entity in entities_list:
        entity_name = ' '.join(entity)
        try:
            if entity_asked == 'director':
                entity_id = director2id[entity_name]
                entity_str = 'di_' + str(entity_id)
                entity_tag = entity_asked
            elif entity_asked == 'genres':
                entity_id = genre2id[entity_name]
                entity_str = 'ge_' + str(entity_id)
                entity_tag = entity_asked

            elif entity_asked == 'country':
                entity_id = country2id[entity_name]
                entity_str = 'co_' + str(entity_id)
                entity_tag = entity_asked
            else:
                entity_id = int(entity_name.split('_')[1])
                entity_str = entity_name
                entity_tag = entity_asked
        except Exception as e:
            return None, None, None

        entity_id_list.append(entity_id)
        entity_id_str_list.append(entity_str)
        entity_tag_list.append(entity_tag)

    if len(entity_id_list) == 0:
        # recognition None
        # print('fail zero')
        return None, None, None
    elif len(entity_id_list) == 1 and 'genres' not in entity_tag_list:
        entity_id_list = entity_id_list[0]
    elif len(entity_id_list) > 1 and 'genres' not in entity_tag_list:
        # print('fail, exclude')

        return None, None, None
    id_str = ' '.join(entity_id_str_list)

    return id_str, entity_id_list, entity_asked


def recommendation(user_id, states, target, recommender, top_k=1):
    target = target
    attributes = {}
    i = 0
    for state in states:
        if state != -1:
            attribute = actions[i]
            if attribute is 'genres':
                attributes[attribute] = state
            else:
                attributes[attribute] = state
        i = i + 1
    # print(attributes)
    # if genres is null
    if states[1] == -1:
        # pop genres to check separately
        no_genres = True
    else:
        no_genres = False
        try:
            target_genres = (attributes.pop('genres'))
            target_genres = [int(genre) for genre in target_genres]
        except Exception as e:
            print('states', states, 'attributes', attributes)
    # get id from database those match all attributes without genres
    # print('states', states, 'attributes', attributes)

    id_list_nogenre = select_by_attributes(attributes)
    if no_genres:
        #print('nogenres', attributes)
        # if genres doesn't in attribute, skip checking genre
        id_list_match_genre = id_list_nogenre
    # check genres
    else:
        id_list_match_genre = []
        for id in id_list_nogenre:
            genre_list = get_genres(id)
            if set(target_genres).issubset(set(genre_list)):
                id_list_match_genre.append(id)
            else:
                pass
    # predict rating of movie matching all attributes
    item_sort, predict = recommender.predict(user_id, id_list_match_genre)

    index_upsort = np.argsort(predict)
    index_downsort = index_upsort[::-1]

    top_k_items = [id_list_match_genre[index] for index in index_downsort[:top_k]]


    return top_k_items


if torch.cuda.is_available():
    print('using cuda')
    device = torch.device('cuda')
else:
    print('using cpu')
    device = torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--prefix")  # data and model prefix
parser.add_argument("--file_name")  # choose model(lstm/bilstm)
parser.add_argument("--boundary_tags")  # add START END tag
args = parser.parse_args()

FILE_PREFIX = args.prefix
file_name = args.file_name

if FILE_PREFIX is None:
    FILE_PREFIX = os.path.expanduser('~/cr_repo/')
if file_name is None:
    file_name = 'simulate/rl_stand_model0.7185185185185186_3.2666666666666666_0.08888888888888889.m'
    # file_name = 'simulate/best_1/rl_model0.7764705882352941_3.223529411764706_0.011764705882352941.m'
    # file_name = 'simulate/best_1/rl_stand_model0.7209302325581395_3.6744186046511627_0.09302325581395349.m'
# file_name = '5turns/po    licy_pretrain_1.5979.pkl'
policy = torch.load(FILE_PREFIX + file_name).to(device)
data_tool = DataTool()
recommender = KNN(FILE_PREFIX, 'recommend/knn_model.m', 'ratings_cleaned.dat')

r_q = -1
r_c = -20
r_rec_fail = -1
max_recreward = 10
max_dialength = 7
device = None


def select_action(i, state_str):
    if i <= max_dialength:
        # select action
        # print('----------------------------------', i)
        state_onehot = state_str
        state_onehot = data_tool.data2onehot(state_onehot)
        state_onehot = data_tool.sparse_2torch(state_onehot)
        # print('state', state_id)
        action = policy.select_best_action(state_onehot, device)

        return action
    else:
        return -1


cache = SimpleCache()
data_id = ['director_id', 'genres_id', 'critic_rating_id', 'country_id', 'audience_rating_id']

app = Flask(__name__)
@app.route('/cr/<answer>')
def get_result(answer):
    print('connection succeed')
    user_input = answer
    print(answer)

    completed_flag = cache.get('completed')

    if user_input == 'hi':
        state_str = ''
        state_id = [-1] * len(data_id)
        i = 0

        data = random.sample(data_sentence_list, 1)
        data = data[0]
        action = select_action(i, state_str)

        entity_asked = actions[action]
        print('data is ', data['origin'])
        five_sentences = data['five_sentences']
        entity2question = {question_sequence[index]: five_sentences[index] for index in range(5)}
        data_answer = entity2question[entity_asked]
        # data_answer_word = [id2word[word] for word in data_answer]
        print(data_answer)

        cache.set('data', data)
        cache.set('action', action)
        cache.set('state_id', state_id)
        cache.set('state_str', state_str)
        cache.set('completed', False)
        cache.set('i', i)

        # return 'you are asked:{},data is:{}'.format(entity_asked, data_answer)
        return 'data is {},\n\n question is {}?'.format(data['origin'], data_answer)
    elif completed_flag is False:
        data = cache.get('data')
        action = cache.get('action')
        state_id = cache.get('state_id')
        state_str = cache.get('state_str')
        i = cache.get('i')

        five_sentences = data['five_sentences']
        entity2question = {question_sequence[index]: five_sentences[index] for index in range(5)}

        i = i + 1
        id_str, entity_id, entity_tag = get_bf_result(action, user_input)
        if entity_id is None:
            print('fail')
            return 'fail'

        state_id[actions.index(entity_tag)] = entity_id
        state_str = state_str + ' ' + id_str

        action = select_action(i, state_str)

        cache.set('data', data)
        cache.set('action', action)
        cache.set('state_id', state_id)
        cache.set('state_str', state_str)
        cache.set('completed', False)
        cache.set('i', i)

        if action in range(5):
            entity_asked = actions[action]
            # print('you are asked:', entity_asked)
            data_answer = entity2question[entity_asked]
            # data_answer_word = [id2word[word] for word in data_answer]
            print('data is ', data_answer)

            return data_answer + '?'
        else:
            user = data['user']
            movie = data['movie']
            top_k = recommendation(user, state_id, movie, recommender)
            print('answer is ', movie)
            print('Do you like ', top_k)

            cache.set('completed', True)

            return 'Result:{},Target is:{}'.format(top_k, movie)
    else:
        print('thank you')
        cache.set('completed', False)
        return 'thank you'


if __name__ == '__main__':
    app.debug = True
    app.run()
