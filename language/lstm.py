"""
LSTM

Input: batch_size x max_text_length (tokenized questions)
Output: batch_size x lstm_hidden_size (question embedding)

Details:
Tokenized text are first word-embedded (300-D), then passed to
2-layer LSTM, where each cell has is 1024-D. For each text,
output the hidden state of the last non-null token.
"""

from __future__ import print_function

import json

import torch
import torch.nn as nn
from torch.autograd import Variable


class Encoder(nn.Module):

    def __init__(self,
                 token_to_idx,
                 word_embedding_dim=300,
                 text_embed_size=1024,
                 metadata_file='./templates/metadata_fsm.json',
                 linear_hidden_size=256,
                 linear_dropout_rate=0):

        super(Encoder, self).__init__()

        # LSTM (shared)
        self.lstm = LSTM(
            token_to_idx=token_to_idx,
            word_embedding_dim=word_embedding_dim,
            lstm_hidden_size=text_embed_size)

        # classifiers (not shared)
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        self.classifier_names = []
        for idx, (key, val) in enumerate(self.metadata.items()):
            num_val = len(val.items())
            classifier_name = key
            self.classifier_names.append(classifier_name)
            setattr(
                self, classifier_name,
                nn.Sequential(
                    fc_block(text_embed_size, linear_hidden_size,
                             linear_dropout_rate),
                    nn.Linear(linear_hidden_size, num_val)))

    def forward(self, text):

        # LSTM (shared)
        # Input: batch_size x max_text_length
        # Output: batch_size x text_embed_size
        text_embedding = self.lstm(text)

        # classifiers (not shared)
        output = []
        for classifier_name in self.classifier_names:
            classifier = getattr(self, classifier_name)
            output.append(classifier(text_embedding))

        return output


class LSTM(nn.Module):

    def __init__(self,
                 token_to_idx,
                 word_embedding_dim=300,
                 lstm_hidden_size=1024,
                 lstm_num_layers=2,
                 lstm_dropout=0):

        super(LSTM, self).__init__()

        # token
        self.token_to_idx = token_to_idx
        self.NULL = token_to_idx['<NULL>']
        self.START = token_to_idx['<START>']
        self.END = token_to_idx['<END>']

        # word embedding
        self.word2vec = nn.Embedding(
            num_embeddings=len(token_to_idx), embedding_dim=word_embedding_dim)

        # LSTM
        self.rnn = nn.LSTM(
            input_size=word_embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bias=True,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=False)

    def forward(self, x):

        batch_size, max_text_length = x.size()

        # Find the last non-null element in each sequence, store in idx
        idx = torch.LongTensor(batch_size).fill_(max_text_length - 1)
        x_cpu = x.data.cpu()
        for text_idx in range(batch_size):
            for token_idx in range(max_text_length - 1):
                if (x_cpu[text_idx, token_idx] != self.NULL
                    ) and x_cpu[text_idx, token_idx + 1] == self.NULL:  # noqa
                    idx[text_idx] = token_idx
                    break
        idx = idx.type_as(x.data).long()
        idx = Variable(idx, requires_grad=False)

        # reduce memory access time
        self.rnn.flatten_parameters()

        # hs: all hidden states
        #      [batch_size x max_text_length x hidden_size]
        # h_n: [2 x batch_size x hidden_size]
        # c_n: [2 x batch_size x hidden_size]
        hidden_states, (_, _) = self.rnn(self.word2vec(x))

        idx = idx.view(batch_size, 1, 1).expand(batch_size, 1,
                                                hidden_states.size(2))
        hidden_size = hidden_states.size(2)

        # only retrieve the hidden state of the last non-null element
        # [batch_size x 1 x hidden_size]
        hidden_state_at_last_token = hidden_states.gather(1, idx)

        # [batch_size x hidden_size]
        hidden_state_at_last_token = hidden_state_at_last_token.view(
            batch_size, hidden_size)

        return hidden_state_at_last_token


class fc_block(nn.Module):

    def __init__(self, inplanes, planes, drop_rate=0.15):
        super(fc_block, self).__init__()
        self.fc = nn.Linear(inplanes, planes)
        self.bn = nn.BatchNorm1d(planes)
        if drop_rate > 0:
            self.dropout = nn.Dropout(drop_rate)
        self.relu = nn.ReLU(inplace=True)
        self.drop_rate = drop_rate

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        if self.drop_rate > 0:
            x = self.dropout(x)
        x = self.relu(x)
        return x


def main():
    """ Test Code """
    # ################### LSTM #########################
    question_token_to_idx = {
        ".": 4,
        "missing": 34,
        "large": 28,
        "is": 26,
        "cubes": 19,
        "cylinder": 21,
        "what": 54,
        "<START>": 1,
        "green": 24,
        "<END>": 2,
        "object": 35,
        "things": 51,
        "<UNK>": 3,
        "matte": 31,
        "rubber": 41,
        "tiny": 52,
        "yellow": 55,
        "red": 40,
        "visible": 53,
        "color": 17,
        "size": 44,
        "balls": 11,
        "the": 48,
        "any": 8,
        "blocks": 14,
        "ball": 10,
        "a": 6,
        "it": 27,
        "an": 7,
        "one": 38,
        "purple": 39,
        "how": 25,
        "thing": 50,
        "?": 5,
        "objects": 36,
        "blue": 15,
        "block": 13,
        "small": 45,
        "shiny": 43,
        "material": 30,
        "cylinders": 22,
        "<NULL>": 0,
        "many": 29,
        "of": 37,
        "cube": 18,
        "metallic": 33,
        "gray": 23,
        "brown": 16,
        "spheres": 47,
        "there": 49,
        "sphere": 46,
        "shape": 42,
        "are": 9,
        "metal": 32,
        "cyan": 20,
        "big": 12
    },

    batch_size = 64
    print('batch size:', batch_size)

    # questions=torch.ones(batch_size, 15, dtype=torch.long)
    questions = torch.randint(0, 10, (batch_size, 15), dtype=torch.long)
    print('intput size:', questions.size())

    lstm = LSTM(token_to_idx=question_token_to_idx[0])
    output = lstm(questions)
    print('output size:', output.size())

    # ################### Language Encoder #########################

    encoder = Encoder(
        token_to_idx=question_token_to_idx[0],
        metadata_file='./templates/metadata_fsm.json')
    output = encoder(questions)
    print('output length:', len(output))
    for classifier in output:
        print('classifier.size():', classifier.size())


if __name__ == '__main__':
    main()
