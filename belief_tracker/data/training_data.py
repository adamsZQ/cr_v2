import json


def get_training_data(file_prefix, data_path):
    data = []
    tags = []
    with open(file_prefix + data_path) as d:
        for line in d:
            data_json = json.loads(line)
            sentence = data_json['key'].split()
            data.append(sentence)

            # TODO multi-tags
            tag = data_json['value']
            tags.append(tag)
    return data, tags


if __name__ == '__main__':
    predix = '/path/bt'
    data, tags = get_training_data(predix)
    print(len(data))
    print(len(tags))