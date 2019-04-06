import json

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer


class DataTool:
    def __init__(self):
        self.__build_vectorizer()

    def __build_vectorizer(self):
        data_list = []
        str_list = []
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
        v.fit_transform(str_list)

        self.v = v

    def data2onehot(self, str):
        str_array = [str]
        return self.v.transform(str_array)

    def sparse_2torch(self, state):
        # 通过coo格式转化为torch稀疏矩阵
        sparse_matrix = state
        X_coo = sparse_matrix.tocoo()
        values = X_coo.data
        indices = np.vstack((X_coo.row, X_coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = X_coo.shape
        X_sparse = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()

        return X_sparse


if __name__ == '__main__':
    data_tool = DataTool()
    aa = data_tool.data2onehot(' ge_1')
    aa = data_tool.sparse_2torch(aa)
    print(data_tool.sparse_2torch(aa))
    print()