import json
import random

with open('/home/next/cr_repo/entity_id/director_id.json', 'r') as f:
    director2id = json.load(f)
id2director = {value: key for key, value in director2id.items()}
with open('/home/next/cr_repo/entity_id/country_id.json', 'r') as f:
    country2id = json.load(f)
id2country = {value: key for key, value in country2id.items()}
with open('/home/next/cr_repo/entity_id/genre_id.json', 'r') as f:
    genre2id = json.load(f)
id2genre = {value: key for key, value in genre2id.items()}

data_sentence_list = []
with open('/home/next/cr_repo/movie_rating', 'r') as m:
    for line in m:
        data_json = json.loads(line)
        data_sentence = {}
        five_sentences = ['which country do you like?', 'which director do you like',
                          'what audience rating do you like?',
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


actions = ['director', 'genres', 'critic_rating', 'country', 'audience_rating', 'recommendation']
# question sequence in training data
question_sequence = ['country', 'director', 'audience_rating', 'critic_rating', 'genres']
data = random.sample(data_sentence_list, 1)
data = data[0]
action = 0

entity_asked = actions[action]
print('you are asked:', entity_asked)
five_sentences = data['five_sentences']
entity2question = {question_sequence[index]: five_sentences[index] for index in range(5)}
data_answer = entity2question[entity_asked]
# data_answer_word = [id2word[word] for word in data_answer]
print('data is ', data_sentence['origin'])