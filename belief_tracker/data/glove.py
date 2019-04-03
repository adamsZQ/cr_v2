import json
import os

import numpy as np

import h5py as h5py


START_TAG = "<START>"
STOP_TAG = "<STOP>"
PADDING_TAG = "<PAD>"
UNK_TAG = '<UNK>'


def get_unk_token():
    """glove doesn't have <unk> token someone calculates the average weight of data as unk"""
    with open('data/unk_file', 'r') as u:
        unk_str = u.read()
        unk = np.array(unk_str.split())
        unk = unk.astype(np.float)
    return unk
    # print(unk)


class Glove_Embeddings():

    def __init__(self, file_prefix, data_path):
        self.file_prefix = file_prefix
        self.data_path = data_path

    def get_glove_embeddings(self):
        # get pretrained glove embeddings
        f = h5py.File(self.file_prefix + 'bf/embeddings/glove.840B.300d.h5', 'r')
        embeddings = f['embedding']
        words_flatten = f['words_flatten']

        return words_flatten, embeddings

    def words_expansion(self):
        task_vocab = []
        task_tags = []
        task_embeddings = []

        # get all words from dialog data
        with open(self.file_prefix + self.data_path) as d:
            for line in d:
                data_json = json.loads(line)
                sentence = data_json['key'].split()
                tags = data_json['value']
                for voca in sentence:
                    if voca not in task_vocab:
                        task_vocab.append(voca)

                if tags not in task_tags:
                    task_tags.append(tags)

        # add start, stop, padding and unk
        task_tags.append(START_TAG)
        task_tags.append(STOP_TAG)
        # task_tags.append(PADDING_TAG)
        task_tags.append(UNK_TAG)
        task_vocab.append(START_TAG)
        task_vocab.append(STOP_TAG)
        # task_vocab.append(PADDING_TAG)
        task_vocab.append(UNK_TAG)
        self.task_word2id = {word: idx for idx, word in enumerate(task_vocab)}
        self.task_id2word = {idx: word for idx, word in enumerate(task_vocab)}
        self.task_tag2id = {tag: idx for idx, tag in enumerate(task_tags)}
        self.task_id2tag = {idx: tag for idx, tag in enumerate(task_tags)}

        words_flatten, embeddings = self.get_glove_embeddings()
        embeddings = embeddings[:]
        words_flatten = str(words_flatten.value, encoding='utf-8').split('\n')

        for word in task_vocab:
            if word in words_flatten:
                task_embeddings.append(embeddings[words_flatten.index(word)])
            elif word is START_TAG:
                # print('start', )
                task_embeddings.append(np.random.rand(len(embeddings[0])))
            elif word is STOP_TAG:
                task_embeddings.append(np.random.rand(len(embeddings[0])))
            else :
                """set oov = random, and fine-tune when training  """
                # print(word)
                task_embeddings.append(np.random.rand(len(embeddings[0])))

        self.task_embeddings = task_embeddings


if __name__ == '__main__':
    glove_embeddings = Glove_Embeddings('/path/bt')
    glove_embeddings.words_expansion()
    task_embeddings = glove_embeddings.task_embeddings
    print(glove_embeddings.task_word2id)
    print(glove_embeddings.task_tag2id)
    print(task_embeddings)