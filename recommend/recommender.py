import json

import numpy

from recommend.knn_recommend.knn import KNN

from tools.sql_tool import select_by_attributes


class Recommender:
    '''
        model: 'knn' or 'fm' (string)
    '''
    def __init__(self, model, repo_prefix, model_path, data_path):
        if model == 'knn':
            self.model = self.__load_knn(repo_prefix, model_path, data_path)
        else:
            pass

        self.repo_prefix = repo_prefix
        self.rec_prefix = 'recommend/'
        # self.attribute_file = 'data/id_attributes.h5'
        self.entity_file = 'data/entity.dat'
        self.movie_rating_file = 'data/'
        # self.id_list, self.attributes_matrix  = self.__load_attributes()

    def __load_entity(self,file_prefix,data_path):
        genres_list = []
        entity_list = []
        with open(file_prefix + data_path, 'r') as f:
            for line in f:
                line = json.loads(line)
                entity_list.append(line)
                genres_list.append(set(line['genres'].split('|')))
        self.entity_list = entity_list
        self.genres_list = genres_list

    def __load_knn(self, file_prefix, model_path, data_path):
        model_loaded = KNN(file_prefix, model_path, data_path)
        return model_loaded

    # def __load_attributes(self):
    #     if os.path.exists(self.repo_prefix + self.rec_prefix):
    #         f = h5py.File(self.repo_prefix + self.rec_prefix + self.attribute_file, 'r')
    #         id_list = f['id']
    #         attributes_matrix = f['attributes']
    #
    #         return id_list, attributes_matrix
    #     else:
    #         # get attributes
    #         entity_file_path = self.repo_prefix + self.rec_prefix + self.entity_file
    #         id_list = []
    #         attribute_list = []
    #         with open(entity_file_path, 'r') as f:
    #             for line in f:
    #                 line = json.loads(line)
    #                 id = line.pop['id']
    #                 id_list.append(id)
    #                 attribute_list.append(line)

    '''
        state_list: slots distribution list, N-gram numpy array 
        id2tag_list: id to tag for each slot 
        state_sequence: example: ['director', 'genres', 'critic_rating', 'country', 'audience_rating'] 
    '''
    def recommend(self, user_id, state_list, id2tag_list, state_sequence, top_k=1, threshold=0.7):
        attributes = {}
        genres_exsit = False
        genres = []
        for state, id2tag, state_entity in zip(state_list, id2tag_list, state_sequence):
            if state_entity != 'genres':
                attributes[state_entity] = id2tag[numpy.argmax(state)]
            elif state_entity == 'genres':
                genres_exsit = True
                i = 0
                for s in state:
                    # TODO how to seperate different genres the user referred
                    if s > threshold:
                        target_genres = id2tag[i]

                    i = i + 1

        item_nogenre = select_by_attributes(attributes)

        if ~genres_exsit:
            # if genres doesn't in attribute, skip checking genre
            id_list_match_genre = item_nogenre
        # check genres
        else:
            id_list_match_genre = []
            for item_id in item_nogenre:
                genre_list = self.genres_list[item_id]
                # genres of item must include target genres
                if set(target_genres).issubset(set(genre_list)):
                    id_list_match_genre.append(id)

        item_sort, predict = self.model.predict(user_id, id_list_match_genre)

        index_upsort = numpy.np.argsort(predict)
        index_downsort = index_upsort[::-1]

        top_k_items = [id_list_match_genre[index] for index in index_downsort[:top_k]]

        return top_k_items
