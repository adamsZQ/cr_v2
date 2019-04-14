import os


from belief_tracker.BiLSTM_CRF_nobatch import BiLSTM_CRF, bilstm_train
from belief_tracker.data.glove import Glove_Embeddings

import argparse
import numpy as np

import torch
from torch import nn

from belief_tracker.data.training_data import get_simulate_data

START_TAG = "<START>"
STOP_TAG = "<STOP>"
PADDING_TAG = "<PAD>"
UNK_TAG = '<UNK>'


# sentences - > padded index sequence
def prepare_sequence(sentences, item2id, boundary_tags=False):
    sentences_idx = []
    for sentence in sentences:
        sentence_idx = []
        if boundary_tags:
            sentence_idx.append(item2id[START_TAG])
            for w in sentence:
                sentence_idx.append(item2id[w])
            sentence_idx.append(item2id[STOP_TAG])

            sentences_idx.append(sentence_idx)
        else:
            sentences_idx.append([item2id[w] for w in sentence])

    return sentences_idx


if __name__ == '__main__':
    # add parser to get prefix
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix") # data and model prefix
    parser.add_argument("--model") # choose model for different slots
    parser.add_argument("--boundary_tags") # add START END tags
    args = parser.parse_args()

    HIDDEN_DIM = 20
    FILE_PREFIX = args.prefix
    model_type = args.model
    bf_prefix = 'bf/'
    boundary_tags = args.boundary_tags
    if boundary_tags is "True":
        boundary_tags = True
    else:
        boundary_tags = False

    if torch.cuda.is_available():
        print('using cuda')
        device = torch.device('cuda')
    else:
        print('using cpu')
        device = torch.device('cpu')

    if FILE_PREFIX is None:
        FILE_PREFIX = '~/cr_repo/'
    if model_type is None:
        model_type = 'test2'
    if boundary_tags is None:
        boundary_tags = False

    FILE_PREFIX = os.path.expanduser(FILE_PREFIX)

    # get training data
    data_path = bf_prefix + model_type + '/training_data'
    sentences_data, tag_data, user_list, movie_list= (get_simulate_data(FILE_PREFIX, data_path))

    # get word embeddings
    glove_embeddings = Glove_Embeddings(FILE_PREFIX, data_path)
    glove_embeddings.words_expansion()
    word_embeddings = glove_embeddings.task_embeddings
    word2id = glove_embeddings.task_word2id
    tag2id = glove_embeddings.task_tag2id

    # sentence data -> index
    sentences_prepared = prepare_sequence(sentences_data, word2id, boundary_tags)
    tag_prepared = prepare_sequence(tag_data, tag2id, boundary_tags)

    # initialize embedding, fine-tune when training
    word_embeds = nn.Embedding.from_pretrained(torch.from_numpy(np.array(word_embeddings)), freeze=False)

    model_prefix = FILE_PREFIX + bf_prefix + model_type + '/'

    bilstm_train(word2id,
                 tag2id,
                 word_embeddings,
                 word_embeds,
                 device,
                 model_prefix,
                 sentences_prepared,
                 tag_prepared, )
