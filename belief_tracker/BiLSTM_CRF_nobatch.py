# -*- coding: utf-8 -*-
import argparse
import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import sys

from tools.save_model import torch_save_model

sys.path.append('..')
sys.path.append(os.path.join(os.getcwd() + '/'))

torch.manual_seed(1)


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def val(model, word_embeds, device, X_val, y_val):
    predict_list = []
    target_list = []
    for sentence, tags in zip(X_val,y_val):
        sentence = torch.tensor(sentence).long().to(device)
        predict = model(word_embeds, sentence)
        predict_list.append(predict[1])
        target_list.append(tags)

    binarizer = MultiLabelBinarizer()
    binarizer.fit_transform([[x for x in range(model.tagset_size)]])
    target_list = binarizer.transform(target_list)
    predict_list = binarizer.transform(predict_list)

    accuracy = accuracy_score(target_list, predict_list)
    f1 = f1_score(target_list, predict_list, average="samples")
    precision = precision_score(target_list, predict_list, average="samples")
    recall = recall_score(target_list, predict_list, average="samples")

    return accuracy, precision, recall, f1

#####################################################################
# Create model


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        # self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

        self.embedding = None

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.).to(device_enable)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        # print('alpha',alpha)
        return alpha

    def _get_lstm_features(self, word_embeds, sentence):
        self.hidden = self.init_hidden()
        embeds = word_embeds(sentence).float().view(1, len(sentence), -1)
        # torch.unsqueeze(embeds,0)
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        # print('lstm-feats',lstm_feats)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1).to(device_enable)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(device_enable), tags])
        for i, feat in enumerate(feats):

            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, word_embeds, sentence, tags):
        feats = self._get_lstm_features(word_embeds, sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    '''
        tag_seq:[0,0,3,0,0]
    '''
    def forward(self, word_embeds, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(word_embeds, sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


START_TAG = "<START>"
STOP_TAG = "<STOP>"
PADDING_TAG = "<PAD>"
UNK_TAG = '<UNK>'


#####################################################################
# Run training
def bilstm_train(word2id,
          tag2id,
          word_embeddings,
          word_embeds,
          device,
          model_prefix,
          sentences_prepared,
          tag_prepared,
          HIDDEN_DIM=4,):

    global device_enable
    device_enable = device
    # model =load_m(model_prefix + 'bilstm_crf_0.0789.pkl')
    model = BiLSTM_CRF(len(word2id), tag2id, word_embeddings[0].size, HIDDEN_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    word_embeds = word_embeds.to(device)
    # word_embeds = model.embedding
    X_train, X_test, y_train, y_test = train_test_split(sentences_prepared, tag_prepared, test_size=0.99, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    print(len(X_train))
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

    epoch = 1000
    best_loss = 1e-1
    model_prefix = model_prefix
    file_name = 'bilstm_crf'
    for num_epochs in range(epoch):
        for sentence, tags in zip(X_train, y_train):

            # Step 3. Run our forward pass.
            sentence = torch.tensor(sentence).long()
            # torch.unsqueeze(sentence, 0)
            tags = torch.tensor(tags).long().to(device)
            # with torch.autograd.profiler.profile() as prof:

            loss = model.neg_log_likelihood(word_embeds, sentence, tags)

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

            # model.embedding = word_embeds

            best_loss = torch_save_model(model, model_prefix, file_name, 1 - f1, best_loss)
            torch_save_model(word_embeds.weight, model_prefix, 'embedding', enforcement=True)

    torch_save_model(model_prefix, file_name, enforcement=True)


def load_model(model_path):
    model = torch.load(model_path)
    # freeze model parameters
    for parameter in model.parameters():
        parameter.requires_grad = False

    return model

