import os

import torch.optim as optim

import sys
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from belief_tracker.data.glove import Glove_Embeddings
from belief_tracker.data.training_data import get_training_data
from belief_tracker.lstm import FacetTracker
from tools.save_model import torch_save_model

import argparse
import numpy as np

import torch
from torch import nn


def val(model, word_embeds, device, X_val, y_val):
    predict_list = []
    target_list = []
    for sentence, tags in zip(X_val,y_val):
        sentence = torch.tensor(sentence).long().to(device)
        tags = torch.tensor(tags).unsqueeze(0).long().to(device)

        lstm_feats, lstm_sofmax = model(word_embeds, sentence)

        lstm_last = lstm_sofmax[-1].unsqueeze(0)
        for i in range(tags.shape[0]):
            tag = tags[i].unsqueeze(0)
            predict = torch.argmax(lstm_last[-1].unsqueeze(0), dim=1)
            predict_list.append(predict.tolist())
            target_list.append([tag])

    binarizer = MultiLabelBinarizer()
    binarizer.fit_transform([[x for x in range(model.output_size)]])
    target_list = binarizer.transform(target_list)
    predict_list = binarizer.transform(predict_list)

    accuracy = accuracy_score(target_list, predict_list)
    f1 = f1_score(target_list, predict_list, average="samples")
    precision = precision_score(target_list, predict_list, average="samples")
    recall = recall_score(target_list, predict_list, average="samples")

    return accuracy, precision, recall, f1


def lstm_train(word2id,
          tag2id,
          word_embeddings,
          word_embeds,
          device,
          model_prefix,
          sentences_prepared,
          tag_prepared,
          HIDDEN_DIM=4,):

    model = FacetTracker(tag2id, HIDDEN_DIM,word_embeddings[0].size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    word_embeds = word_embeds.to(device)
    # save word embeds
    model.embedding = word_embeds

    tag_prepared = np.reshape(tag_prepared, (-1)).tolist()
    # control total number of data
    X_train, X_test, y_train, y_test = train_test_split(sentences_prepared, tag_prepared, test_size=0.0, random_state=2,shuffle=True)
    # split train test
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0, shuffle=True)
    # split val test
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

    epoch = 1000
    best_loss = 1e-1
    model_prefix = model_prefix
    file_name = 'lstm'
    for num_epochs in range(epoch):
        # for step
        for sentence, tags in zip(X_train, y_train):
            sentence = torch.tensor(sentence).long().to(device)
            tags = torch.tensor(tags).unsqueeze(0).long().to(device)

            lstm_feats, lstm_sofmax = model(word_embeds, sentence)

            # for genres need multi-tags update
            lstm_last = lstm_feats[-1].unsqueeze(0)
            for i in range(tags.shape[0]):
                tag = tags[i].unsqueeze(0)

                # cross entropy already has softmax
                loss = criterion(lstm_last, tag)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if num_epochs % 1 == 0:
            accuracy, precision, recall, f1 = val(model, word_embeds, device, X_val, y_val)

            print('Epoch[{}/{}]'.format(num_epochs, epoch) + 'loss: {:.6f}'.format(
                loss.item()) +
                  'accuracy_score: {:.6f}'.format(accuracy) +
                  'precision_score: {:.6f}'.format(precision) +
                  'recall_score: {:.6f}'.format(recall) +
                  'f1_score: {:.6f}'.format(f1))
            sys.stdout.flush()

            # save word embedding
            model.embedding = word_embeds

            best_loss = torch_save_model(model, model_prefix, file_name, 1 - f1, best_loss)

    torch_save_model(model_prefix, file_name, enforcement=True)


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

    HIDDEN_DIM = 4
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
        model_type = 'test3'
    if boundary_tags is None:
        boundary_tags = True

    FILE_PREFIX = os.path.expanduser(FILE_PREFIX)

    # get training data
    data_path = bf_prefix + model_type + '/training_data'
    sentences_data, tag_data = (get_training_data(FILE_PREFIX, data_path))

    # get word embeddings
    glove_embeddings = Glove_Embeddings(FILE_PREFIX, data_path)
    glove_embeddings.words_expansion()
    word_embeddings = glove_embeddings.task_embeddings
    word2id = glove_embeddings.task_word2id
    tag2id = glove_embeddings.task_tag2id

    # sentence data -> index
    sentences_prepared = prepare_sequence(sentences_data, word2id, boundary_tags)
    tag_prepared = prepare_sequence([tag_data], tag2id, boundary_tags)

    # initialize embedding, fine-tune when training
    word_embeds = nn.Embedding.from_pretrained(torch.from_numpy(np.array(word_embeddings)), freeze=False)

    model_prefix = FILE_PREFIX + bf_prefix + model_type + '/'

    lstm_train(word2id,
                 tag2id,
                 word_embeddings,
                 word_embeds,
                 device,
                 model_prefix,
                 sentences_prepared,
                 tag_prepared, )
