import json

from tools.simple_tools import chunks

with open('/home/next/cr_repo/entity2id.dat', 'r') as f:
    entity2id = json.load(f)

movie_list = []
director_dict = {}
country_dict = {}
genre_dict = {}
with open('/home/next/cr_repo/movie_data_cleaned2', 'r') as w:
    for line in w:
        movie = json.loads(line)
        director = movie['director']
        country = movie['country']
        genre_list = movie['genres'].split('|')

        director_dict[director] = entity2id[director]
        country_dict[country] = entity2id[country]
        for genre in genre_list:
            genre_dict[genre] = int(entity2id[genre])

with open('/home/next/cr_repo/entity_id/director_id.json', 'w') as c:
    c.write(json.dumps(director_dict))

with open('/home/next/cr_repo/entity_id/country_id.json', 'w') as v:
    v.write(json.dumps(country_dict))

with open('/home/next/cr_repo/entity_id/genre_id.json', 'w') as b:
    b.write(json.dumps(genre_dict))
