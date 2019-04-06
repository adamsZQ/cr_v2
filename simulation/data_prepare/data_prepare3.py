import json

data_named_list = []
with open('/home/next/cr_repo/movie_data_cleaned2', 'r') as t:
    for line in t:
        line = json.loads(line)
        data_named_list.append(line)


data_id_list = []
with open('/home/next/cr_repo/entity.dat') as w:
    for line in w:
        data_id = json.loads(line)
        data_id_list.append(data_id)


entity2id = {}
for data_named, data_id in zip(data_named_list, data_id_list):
    director_named = data_named['director']
    country_named = data_named['country']
    genres_named = data_named['genres'].split('|')

    director_id = data_id['director']
    country_id = data_id['country']
    genres_id = data_id['genres'].split('|')

    if director_named not in entity2id:
        entity2id[director_named] = director_id
    if country_named not in entity2id:
        entity2id[country_named] = country_id

    for genres_named, genres_id in zip(genres_named, genres_id):
        if genres_named not in entity2id:
            entity2id[genres_named] = genres_id


with open('/home/next/cr_repo/entity2id.dat', 'w') as t:
    t.write(json.dumps(entity2id))