import os

import torch
import torch.nn as nn

import sys


sys.path.append(os.path.join(os.getcwd() + '/'))


class FacetTracker(nn.Module):
    def __init__(self, tag_to_ix, hidden_size, embedding_dim):
        super(FacetTracker, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = len(tag_to_ix)

        # batch * seq_len * input_size x * 4 * 4287
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=1, batch_first=True)

        # hidden->slots
        self.out = nn.Linear(self.hidden_size, self.output_size, True)

        # softmax -> probability
        self.softmax = nn.Softmax(dim=1)

        # after training, save embedding
        self.embedding = None

    def forward(self, word_embeds, sentence):
        # sentence -> words embedding
        embeds = word_embeds(sentence).float().view(1, len(sentence), -1)
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_out = lstm_out.view(len(sentence), self.hidden_size)
        lstm_feats = self.out(lstm_out)
        lstm_sofmax = self.softmax(lstm_feats)
        return lstm_feats, lstm_sofmax


def load_model(model_path):
    model = torch.load(model_path)
    # freeze model parameters
    for parameter in model.parameters():
        parameter.requires_grad = False

    return model






