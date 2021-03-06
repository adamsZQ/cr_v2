import argparse
import json
import os
import sys
import time
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
X_train, X_test, y_train, y_test = train_test_split(data_zipped, tag_chunk, test_size=0.9, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=8)
print(len(X_train))
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=2)

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


def get_bf_result(action, entity2question, word_embeds):
    entity_asked = actions[action]
    question = entity2question[entity_asked]
    sentence = torch.tensor(question).long().to(device)
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


def max_entropy_select_action(state, question_turn):
    for i in range(question_turn):
        if state[i] == -1:
            return i
    return 5


def max_entropy_simulate(entropy_turn, recommender, max_dialength, max_recreward, r_rec_fail, device, r_c, r_q):
    reward_list = []
    conversation_turn_num = []
    correct_num = 0
    quit_num = 0
    for data in X_test:
        five_sentences = data['five_sentences']
        user = data['user']
        movie = data['movie']
        entity2question = {question_sequence[i]: five_sentences[i] for i in range(5)}

        data_id = ['director_id', 'genres_id', 'critic_rating_id', 'country_id', 'audience_rating_id']
        state_str = ''
        state_id = [-1] * len(data_id)
        reward = 0

        prob = 1
        e_t = np.random.choice([entropy_turn, entropy_turn - 1], p=[prob, 1-prob])
        # print(max_dialength)
        for i in range(max_dialength):

            action = max_entropy_select_action(state_id, e_t)

            # if action asks question
            if action in range(5):
                # if max_dialog length still asking question, give r_q
                if i == max_dialength - 1:
                    # print('over length')
                    reward = r_q
                    quit_num = quit_num + 1
                    break
                else:
                    # print('ask question')
                    if state_id[action] == -1:
                        # get result from belief_tracker, result is a triplet (question, user, movie)
                        id_str, entity_id, entity_tag = get_bf_result(action, entity2question, word_embeds)
                        if entity_id is None:
                            reward = r_q
                            quit_num = quit_num + 1
                            break
                        else:
                            state_id[actions.index(entity_tag)] = entity_id
                            state_str = state_str + ' ' + id_str
                    reward = r_c
            # if action is recommendation
            elif action == 5:
                # reward = max_recreward
                if recommendation(user, state_id, movie, recommender):
                    # recommend successfully
                    # print('recommend success')
                    reward = max_recreward
                    correct_num = correct_num + 1
                else:
                    # fail
                    # print('recommend fail')
                    reward = r_rec_fail
                break
            else:
                # print('wrong action')
                policy.rewards.append(r_q)
                break

            # append reward
            # print('reward',reward)
            policy.rewards.append(reward)
            reward_list.append(reward)
        #print('reward', reward)
        policy.rewards.append(reward)
        reward_list.append(reward)

        # append conversation turn num
        conversation_turn_num.append(i + 1)

    ave_reward = np.mean(reward_list)
    ave_conv = np.mean(conversation_turn_num)
    accuracy = float(correct_num) / len(X_val)
    quit_rating = float(quit_num) / len(X_val)

    return ave_reward, ave_conv, accuracy, quit_rating


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

    # print('result', list(zip(item_sort, predict)))

    if int(target) in top_k_items:
        #print('succeed!')
        return True
    else:
        #print('fail!')
        return False


if __name__ == '__main__':
    if torch.cuda.is_available():
        print('using cuda')
        device = torch.device('cuda')
    else:
        print('using cpu')
        device = torch.device('cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix") # data and model prefix
    parser.add_argument("--file_name") # choose model(lstm/bilstm)
    parser.add_argument("--boundary_tags") # add START END tag
    args = parser.parse_args()

    FILE_PREFIX = args.prefix
    file_name = args.file_name

    if FILE_PREFIX is None:
        FILE_PREFIX = os.path.expanduser('~/cr_repo/')
    if file_name is None:
        file_name = 'simulate/best_1/rl_stand_model0.7209302325581395_3.883720930232558_0.09302325581395349.m'

    # file_name = '5turns/po    licy_pretrain_1.5979.pkl'
    policy = torch.load(FILE_PREFIX+file_name).to(device)
    data_tool = DataTool()
    recommender = KNN(FILE_PREFIX, 'recommend/knn_model.m', 'ratings_cleaned.dat')
    # simulate(policy, recommender, r_q=-1, r_c=0, r_rec_fail=-1, max_recreward=0.1)

    val_ave_reward, val_ave_conv, val_accuracy,  val_quit_rating = max_entropy_simulate(3, recommender, r_q=-1, r_c=0, r_rec_fail=-1, max_recreward=1.2, max_dialength=7, device=None)

    print('val_ave_reward: {:.6f}'.format(val_ave_reward) +
          'val_accuracy_score: {:.6f}'.format(val_accuracy) +
          'val_ave_conversation: {:.6f}'.format(val_ave_conv) +
          'val_quit_rating: {:.6f}'.format(val_quit_rating)
          )

