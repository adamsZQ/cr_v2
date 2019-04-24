import argparse
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn, optim

from belief_tracker.BiLSTM_CRF_nobatch import BiLSTM_CRF, load_model
from belief_tracker.data.glove import Glove_Embeddings
from belief_tracker.data.training_data import get_simulate_data
from belief_tracker.train.bilstm_training import prepare_sequence


def val(model, word_embeds, device, X_val, y_val):
    predict_list = []
    target_list = []
    for sentence, tags in zip(X_val,y_val):
        sentence = torch.tensor(sentence).long().to(device)
        predict = model(word_embeds, sentence)
        for pre in predict[1]:
            predict_list.append(pre)
        for tag in tags:
            target_list.append(tag)

    predict_list = np.reshape(predict_list, (-1, 1)).tolist()
    target_list = np.reshape(target_list, (-1, 1)).tolist()

    accuracy = accuracy_score(target_list, predict_list)
    f1 = f1_score(target_list, predict_list, average="macro")
    precision = precision_score(target_list, predict_list, average="macro")
    recall = recall_score(target_list, predict_list, average="macro")

    return accuracy, precision, recall, f1


# add parser to get prefix
parser = argparse.ArgumentParser()
parser.add_argument("--prefix") # data and model prefix
parser.add_argument("--model") # choose model for different slots
parser.add_argument("--boundary_tags") # add START END tags
args = parser.parse_args()

HIDDEN_DIM = 128
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
    model_type = 'test6'
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

model_prefix = FILE_PREFIX + bf_prefix + model_type + '/'
model_path = '/home/next/cr_repo/bf/test6/bilstm_crf_0.0052.pkl'
bf_model = load_model(model_path)
# TODO load word embedding
embedding_path = '/home/next/cr_repo/bf/test6/embedding0.005154639175257714_enforcement.pkl'
word_embeds_weight = torch.load(embedding_path)
word_embeds = nn.Embedding.from_pretrained(word_embeds_weight, freeze=True)
# word_embeds = model.embedding
X_train, X_test, y_train, y_test = train_test_split(sentences_prepared, tag_prepared, test_size=0.96, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
print(len(X_train))
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1)


accuracy, precision, recall, f1 = val(bf_model, word_embeds, device, X_test, y_test)

print(
      'accuracy_score: {:.6f}'.format(accuracy) +
      'precision_score: {:.6f}'.format(precision) +
      'recall_score: {:.6f}'.format(recall) +
      'f1_score: {:.6f}'.format(f1))