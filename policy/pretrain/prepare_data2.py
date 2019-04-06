
import json
import random

import h5py as h5py
import numpy as np
import torch
from scipy.sparse import hstack, vstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

data_list = []
str_list = []
# director_list = []
# genres_list = []
# genres_single_list = []
# critic_rating_list = []
# country_list = []
# audience_rating_list = []
with open('/home/next/cr_repo/movie_rating', 'r') as f:
    for line in f:
        line = json.loads(line)
        data_list.append(line)

        # print(str(line['user']))
        director = 'di_' + str(line['director'])
        genres = ' '.join(['ge_' + str(genre) for genre in line['genres'].split('|')])
        critic_rating = 'cr_' + str(line['critic_rating'])
        country = 'co_' + str(line['country'])
        audience_rating = 'au_' + str(line['audience_rating'])

        data_str = ' '.join([director, genres, critic_rating, country, audience_rating])
        str_list.append(data_str)

v = CountVectorizer()
a = v.fit_transform(str_list)


def rule_based_action(brunch_num):
    # construct each 2000 data based on max entropy
    state_list = []
    action_list = []
    # TODO before supply, remove static question_sequence
    question_sequence = ['director', 'genres', 'critic_rating', 'country', 'audience_rating']
    # question_sequence = max_entropy_4all('/path/mv/movie_rating', match_all_genres=False)

    '''    
    actions = ['director', 'genres', 'critic_rating', 'country', 'audience_rating', 'recommendation]
    '''

    question_maxlen = len(question_sequence)


    data_slice = random.sample(data_list, 20000)
    for data in data_slice:
        print('data generation starts')
        director = data['director']
        genres = data['genres'].split('|')
        critic_rating = data['critic_rating']
        country = data['country']
        audience_rating \
            = data['audience_rating']

        state_init = ''
        state_list.append(state_init)
        action_list.append(0)

        state_init = state_init + 'di_' + str(director)
        state_list.append(state_init)
        action_list.append(1)

        genres = ' '.join(['ge_' + str(genre) for genre in genres])
        state_init = ' '.join([state_init, genres])
        state_list.append(state_init)
        action_list.append(2)

        state_init = ' '.join([state_init, 'cr_' + str(critic_rating)])
        state_list.append(state_init)
        action_list.append(3)

        state_init = ' '.join([state_init, 'co_' + str(country)])
        state_list.append(state_init)
        action_list.append(4)

        state_init = ' '.join([state_init, 'au_' + str(audience_rating)])
        state_list.append(state_init)
        action_list.append(5)

        # print('------------------------------')
        # print(len(state_list))
        # print(len(action_list))


    # state_list = np.array(state_list).reshape(-1,question_maxlen)
    state_list_trans = v.transform(state_list)
    print(state_list_trans[:50])
    # state_sparse = vstack(state_list)
    state_torch = sparse_2torch(state_list_trans)
    action_list = np.array(action_list).reshape(-1, 1)

    f = h5py.File('/home/next/cr_repo/pre_train/pretrain_data.h5', 'w')
    f.create_dataset(data=state_torch, name='states')
    f.create_dataset(data=action_list, name='actions')


def sparse_2torch(sparse_matrix):
    # 通过coo格式转化为torch稀疏矩阵
    X_coo = sparse_matrix.tocoo()
    values = X_coo.data
    indices = np.vstack((X_coo.row, X_coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = X_coo.shape
    X_sparse = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

    return X_sparse

if __name__ == '__main__':
    rule_based_action(2000)