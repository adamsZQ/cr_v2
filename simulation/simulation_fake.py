import argparse
import json
import os
import sys
import time
import profile
import numpy as np

import torch
from sklearn.model_selection import train_test_split
from torch import optim

from recommend.recommender import Recommender
from tools.data_transfer import DataTool
from tools.sql_tool import select_by_attributes, select_genres, select_all_movie_genres, select_all

data_list = []
director_list = []
genres_list = []
critic_rating_list = []
country_list = []
audience_rating_list = []
with open(os.path.expanduser('~/path/mv/movie_rating'), 'r') as f:
    for line in f:
        line = json.loads(line)
        data_list.append(line)
        director_list.append(line['director'])
        genres_list.append(set(line['genres'].split('|')))
        critic_rating_list.append(line['critic_rating'])
        country_list.append(line['country'])
        audience_rating_list.append(line['audience_rating'])

# get part of datalist
data_list, useless, a, b = train_test_split(data_list, [0] * len(data_list), test_size=0.96, random_state=1)
# data_list, useless, a, b = train_test_split(data_list, [0] * len(data_list), test_size=0.9, random_state=1)
# train test split
trainset, testset, a, b = train_test_split(data_list, [0] * len(data_list), test_size=0.2, random_state=1)
print(len(trainset))
testset, valset, a, b = train_test_split(testset, [0] * len(testset), test_size=0.5, random_state=2)


actions = ['director', 'genres', 'critic_rating', 'country', 'audience_rating', 'recommendation']

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


def simulate(model, recommender, max_dialength=7, max_recreward=50,r_rec_fail=-10, r_c=-1, r_q=-10):
    print('simulate start')
    num_epochs = 10000

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    for epoch in range(num_epochs):
        reward_list = []
        conversation_turn_num = []
        correct_num = 0
        t_start = time.time()
        t_rec = 0
        quit_num = 0

        for data in trainset:
            director_id = data['director']
            genres_id = data['genres'].split('|')
            critic_rating_id = data['critic_rating']
            country_id = data['country']
            audience_rating_id = data['audience_rating']

            director = 'di_' + str(data['director'])
            genres = ' '.join(['ge_' + str(genre) for genre in data['genres'].split('|')])
            critic_rating = 'cr_' + str(data['critic_rating'])
            country = 'co_' + str(data['country'])
            audience_rating = 'au_' + str(data['audience_rating'])
            user = data['user']
            target = data['movie']

            data_str = [director, genres, critic_rating, country, audience_rating]
            data_id = [director_id, genres_id, critic_rating_id, country_id, audience_rating_id]
            state_str = ''
            state_id = [-1] * len(data_id)
            reward = 0

            for i in range(max_dialength):
                # select action
                # print('----------------------------------', i)
                state_onehot = state_str
                state_onehot = data_tool.data2onehot(state_onehot)
                # print('state_onehot', state_onehot)
                # print('state_id', state_id)
                state_onehot = data_tool.sparse_2torch(state_onehot)
                action = model.select_action(state_onehot, device)
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
                            state_id[action] = data_id[action]
                            state_str = state_str + ' ' + data_str[action]
                        reward = r_c
                # if action is recommendation
                elif action == 5:
                    # reward = max_recreward
                    t_rec_start = time.time()
                    if recommendation(user, state_id, target, recommender):
                        # recommend successfully
                        # print('recommend success')
                        reward = max_recreward
                        correct_num = correct_num + 1
                        t_rec_done = time.time()
                        t_rec = t_rec + t_rec_done - t_rec_start
                    else:
                        # fail
                        # print('recommend fail')
                        reward = r_rec_fail
                    break
                else:
                    # print('wrong action')
                    model.rewards.append(r_q)
                    break

                # append reward
                #print('reward',reward)
                model.rewards.append(reward)
                reward_list.append(reward)

            # append reward
            #print('reward', reward)

            model.rewards.append(reward)
            reward_list.append(reward)
            # append conversation turn num
            conversation_turn_num.append(i+1)
            # update policy
            #print('update')
            model.update_policy(optimizer)

        if epoch % 1 == 0:
            print('sequence time:', time.time()-t_start)
            print('rec time:', t_rec)

            train_ave_reward = np.mean(reward_list)
            # ave_reward = np.mean(reward_list)
            ave_conv = np.mean(conversation_turn_num)
            accuracy = float(correct_num) / len(trainset)
            quit_rating = float(quit_num) / len(trainset)

            val_ave_reward, val_ave_conv, val_accuracy, val_quit_rating = val(model, recommender, max_dialength, max_recreward, r_rec_fail, None, r_c, r_q)

            # ave_reward, ave_conv, accuracy = val(model, recommender, max_dialength, max_recreward, r_c, r_q)
            print('Epoch[{}/{}]'.format(epoch, num_epochs) +
                  'train ave_reward: {:.6f}'.format(train_ave_reward) +
                  'accuracy_score: {:.6f}'.format(accuracy) +
                  'ave_conversation: {:.6f}'.format(ave_conv) +
                  'quit_rating: {:.6f}'.format(quit_rating)
            )
            print('val_ave_reward: {:.6f}'.format(val_ave_reward) +
                  'val_accuracy_score: {:.6f}'.format(val_accuracy) +
                  'val_ave_conversation: {:.6f}'.format(val_ave_conv) +
                  'val_quit_rating: {:.6f}'.format(val_quit_rating)
                  )

            sys.stdout.flush()

            if val_accuracy > 0.7:
                print('save model')
                torch.save(model, '/home/next/cr_repo/simulate/rl_model{}_{}_{}.m'.format(val_accuracy, val_ave_conv, val_quit_rating))


def val(model, recommender, max_dialength, max_recreward, r_rec_fail, device, r_c, r_q):
    reward_list = []
    conversation_turn_num = []
    correct_num = 0
    quit_num = 0
    for data in valset:
        director_id = data['director']
        genres_id = data['genres'].split('|')
        critic_rating_id = data['critic_rating']
        country_id = data['country']
        audience_rating_id = data['audience_rating']

        director = 'di_' + str(data['director'])
        genres = ' '.join(['ge_' + str(genre) for genre in data['genres'].split('|')])
        critic_rating = 'cr_' + str(data['critic_rating'])
        country = 'co_' + str(data['country'])
        audience_rating = 'au_' + str(data['audience_rating'])
        user = data['user']
        target = data['movie']

        data_str = [director, genres, critic_rating, country, audience_rating]
        data_id = [director_id, genres_id, critic_rating_id, country_id, audience_rating_id]
        state_str = ''
        state_id = [-1] * len(data_id)
        reward = 0

        for i in range(max_dialength):
            # select action
            # print('----------------------------------', i)
            state_onehot = state_str
            state_onehot = data_tool.data2onehot(state_onehot)
            state_onehot = data_tool.sparse_2torch(state_onehot)
            action = model.select_best_action(state_onehot, device)
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
                    state_id[action] = data_id[action]
                    state_str = state_str + ' ' + data_str[action]
                    reward = r_c
            # if action is recommendation
            elif action == 5:
                # reward = max_recreward
                if recommendation(user, state_id, target, recommender):
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
                model.rewards.append(r_q)
                break

            # append reward
            # print('reward',reward)
            model.rewards.append(reward)
            reward_list.append(reward)
        #print('reward', reward)
        model.rewards.append(reward)
        reward_list.append(reward)

        # append conversation turn num
        conversation_turn_num.append(i + 1)

    ave_reward = np.mean(reward_list)
    ave_conv = np.mean(conversation_turn_num)
    accuracy = float(correct_num) / len(valset)
    quit_rating = float(quit_num) / len(valset)

    return ave_reward, ave_conv, accuracy, quit_rating


def recommendation(user_id, states, target, recommender, top_k=5):
    index = [i for i in range(len(states)) if states[i] == -1]
    len_index = len(index)
    # print(len_index)
    # if len_index == 0:
    #     rating = 0.997187
    # elif len_index == 1:
    #     rating = 0.995071
    # elif len_index == 2:
    #     rating = 0.993133
    # elif len_index == 3:
    #     rating = 0.972280
    # elif len_index == 4:
    #     rating = 0.770491
    # elif len_index == 5:
    #     rating = 0.0

    if len_index == 0:
        rating = 0.889718
    elif len_index == 1:
        rating = 0.859617
    elif len_index == 2:
        rating = 0.836962
    elif len_index == 3:
        rating = 0.755233
    elif len_index == 4:
        rating = 0.341166
    elif len_index == 5:
        rating = 0.0

    success = np.random.choice(2,1,p=[1-rating,rating])

    if success == 1:
        return True
    else:
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
        file_name = 'pre_train/policy_pretrain_5turns_0.2202.pkl'

        # file_name = '5turns/policy_pretrain_1.5979.pkl'
    model = torch.load(FILE_PREFIX + file_name).to(device)

    # recommender = Recommender(FILE_PREFIX, 'model/knn_model.m', 'ratings_cleaned.dat')
    recommender = None
    data_tool = DataTool()
    simulate(model, recommender, r_q=-1, r_c=0, r_rec_fail=-3, max_recreward=3)

    # val_ave_reward, val_ave_conv, val_accuracy, val_quit_rating = val(model, recommender, r_q=-1, r_c=0, r_rec_fail=-1, max_recreward=0.1, max_dialength=7, device=None)
    #
    # print('val_ave_reward: {:.6f}'.format(val_ave_reward) +
    #       'val_accuracy_score: {:.6f}'.format(val_accuracy) +
    #       'val_ave_conversation: {:.6f}'.format(val_ave_conv) +
    #       'val_quit_rating: {:.6f}'.format(val_quit_rating)
    #       )