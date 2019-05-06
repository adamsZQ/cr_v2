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
    model_type = 'test6'
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
model_path = '/home/next/cr_repo/bf/test6/bilstm_crf_0.0052.pkl'
bf_model = load_model(model_path)
# TODO load word embedding
embedding_path = '/home/next/cr_repo/bf/test6/embedding0.005154639175257714_enforcement.pkl'
word_embeds_weight = torch.load(embedding_path)
word_embeds = nn.Embedding.from_pretrained(word_embeds_weight, freeze=True)

# get part of datalist
X_train, X_test, y_train, y_test = train_test_split(data_zipped, tag_chunk, test_size=0.96, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
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


def get_bf_result(action, entity2question, word_embeds):
    entity_asked = actions[action]
    question = entity2question[entity_asked]
    question_word = [id2word[word] for word in question]
    print(question_word)
    # sentence = torch.tensor(question).long().to(device)
    sentence = interact(entity_asked)
    sentence_tensor = torch.tensor(sentence).squeeze(0).long().to(device)

    # print(sentence)
    predict = bf_model(word_embeds, sentence_tensor)
    tags_pred_list = predict[1]

    # sentence_tensor = sentence_tensor.squeeze(0)
    entity_id_str = []
    entity_id = []
    entity_tag = []
    for word, tag in zip(sentence_tensor, tags_pred_list):
        tag_name = id2tag[tag]
        word_name = id2word[word.tolist()]
        if tag_name != 'O':
            tag_entity = tag_name.split('-')[1]

            if entity_asked == tag_entity:
                # if recognition fail, continue
                if '_' not in word_name:
                    # print('bf recognition fail', word_name, tag_entity)
                    continue
                else:
                    entity_id_str.append(word_name)
                    # entity_id_int.append(word_name.split('_')[1])
                    entity_tag.append(tag_entity)

                    entity_id.append(int(word_name.split('_')[1]))
    if len(entity_id) == 0:
        # recognition None
        return None, None, None
    elif len(entity_id) == 1 and 'genres' not in entity_tag:
        entity_id = entity_id[0]
    elif len(entity_id) > 1 and 'genres' not in entity_tag:
        return None, None, None
    id_str = ' '.join(entity_id_str)
    return id_str, entity_id, entity_asked


def interact(entity_asked):
    print('Answer the question about ', entity_asked)
    answer = input("Enter your answer: ")
    answer_index = prepare_sequence([answer.split()], word2id, False)

    return answer_index


def evaluation(policy, recommender, max_dialength, max_recreward, r_rec_fail, device, r_c, r_q):
    reward_list = []
    conversation_turn_num = []
    correct_num = 0
    quit_num = 0
    for data in X_val:
        five_sentences = data['five_sentences']
        user = data['user']
        movie = data['movie']
        entity2question = {question_sequence[i]: five_sentences[i] for i in range(5)}

        data_id = ['director_id', 'genres_id', 'critic_rating_id', 'country_id', 'audience_rating_id']
        state_str = ''
        state_id = [-1] * len(data_id)
        reward = 0

        for i in range(max_dialength):
            # select action
            # print('----------------------------------', i)
            state_onehot = state_str
            state_onehot = data_tool.data2onehot(state_onehot)
            state_onehot = data_tool.sparse_2torch(state_onehot)
            # print('state', state_id)
            action = policy.select_best_action(state_onehot, device)
            # print('state', state)
            #
            # print('action', action)

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
                top_k = recommendation(user, state_id, movie, recommender)
                print('answer is ', movie)
                print('Do you like ', top_k)
                answer = input("Enter your answer: ")

                if str(answer) == '1':
                    reward = max_recreward
                    correct_num = correct_num + 1
                else:
                    reward = r_rec_fail

                # # reward = max_recreward
                # if recommendation(user, state_id, movie, recommender):
                #     # recommend successfully
                #     # print('recommend success')
                #     reward = max_recreward
                #     correct_num = correct_num + 1
                # else:
                #     # fail
                #     # print('recommend fail')
                #     reward = r_rec_fail
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

    return top_k_items
    # if int(target) in top_k_items:
    #     #print('succeed!')
    #     return True
    # else:
    #     #print('fail!')
    #     return False


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
        file_name = 'simulate/rl_stand_model0.7185185185185186_3.2666666666666666_0.08888888888888889.m'
        # file_name = 'simulate/best_1/rl_model0.7764705882352941_3.223529411764706_0.011764705882352941.m'
        # file_name = 'simulate/best_1/rl_stand_model0.7209302325581395_3.6744186046511627_0.09302325581395349.m'
    # file_name = '5turns/po    licy_pretrain_1.5979.pkl'
    policy = torch.load(FILE_PREFIX+file_name).to(device)
    data_tool = DataTool()
    recommender = KNN(FILE_PREFIX, 'recommend/knn_model.m', 'ratings_cleaned.dat')
    evaluation(policy, recommender,  r_q=-1, r_c=-20, r_rec_fail=-1, max_recreward=10, max_dialength=7, device=None)

