import json


def get_bf_training_data(file_prefix, data_path):
    data = []
    tags = []
    with open(file_prefix + data_path) as d:
        for line in d:
            # if len(line) < 3:
            #     continue
            data_json = json.loads(line)
            sentence = data_json['key'].split()
            data.append(sentence)

            tag = data_json['tags']
            tags.append(tag)
    return data, tags


def get_simulate_data(file_prefix, data_path):
    data = []
    tags = []
    user_list = []
    movie_list = []
    with open(file_prefix + data_path) as d:
        for line in d:

            data_json = json.loads(line)
            sentence = data_json['key'].split()
            data.append(sentence)

            tag = data_json['tags']
            tags.append(tag)

            user = data_json['user']
            movie = data_json['movie']
            user_list.append(user)
            movie_list = data_json['movie']
    return data, tags, user_list, movie_list


if __name__ == '__main__':
    predix = '/path/bt'
    data, tags = get_bf_training_data(predix)
    print(len(data))
    print(len(tags))