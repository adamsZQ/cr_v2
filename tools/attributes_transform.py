from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

'''
    transform attributes to one-hot(genres are multi-hot)
    input:dict_list
    output:numpy array
'''
def transform(attribute_list):
    v = DictVectorizer()
    c = CountVectorizer()
    genre_list = []
    for line in attribute_list:
        genres = line.pop('genres').replace('|', ' ')
        genre_list.append(genres)

    attribute_nognres = v.fit_transform(attribute_list)
    genre_transormed_list = c.fit_transform(genre_list)

    attribute_transformed = hstack([attribute_nognres, genre_transormed_list])
