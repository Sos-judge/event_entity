# import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Union  # for type hinting
# import torch.autograd as autograd
# import src.all_models.model_utils


class CDCorefScorer(nn.Module):
    '''
    An abstract class represents a coreference pairwise scorer.
    Inherits Pytorch's Module class.
    '''
    def __init__(self,
                 word_embeds: np.ndarray, word_to_ix: Dict[str, int], vocab_size: int,
                 char_embeds, char_to_ix, char_rep_size, dims, use_mult, use_diff, feature_size
                 ):
        """
        init for CorefScorer object

        :param word_embeds: Word embeddings.
            It is an array with size(|V|, |w|).
            |v| is the length of vocabulary.
            |w| is the length of word embedding and is 300 by default beacause we use glove.6B.300d.txt by default.
            The element word_embeds[i] is the word embedding of the i-th word in vocabulary.
        :param word_to_ix: A lookup dict of word_embeds.
            Key is each word in vocabulary.
            Value is the index of this word's embedding in word_embeds.
            So, word_embeds[word_to_ix["cat"]] is the embedding of word "cat".
        :param vocab_size:  The vocabulary size, that is word_embeds.shape[0].
        :param char_embeds: Char embeddings
            It is an array with size(96, 300).
            96: There are 94 chars in Glove char embeddings file and 2 more special char.
            300: The length of word embedding. It is 300 by default beacause we use glove.6B.300d.txt by default.
            The element char_embeds[i] is the char embedding of the i-th char in vocabulary.
        :param char_to_ix: A lookup dict of char_embedding.
            Key is each char in vocabulary.
            Value is the index of this char's embedding in word_embeds.
            So, char_embeds[char_to_ix["$"]] is the embedding of char "$".
            Length is 96: There are 94 chars in Glove char embeddings file and 2 more special char.
        :param char_rep_size: Hidden size of the character LSTM
        :param dims: A list that holds the layer dimensions
        :param use_mult: a boolean indicates whether to use element-wise multiplication in the input layer
        :param use_diff: a boolean indicates whether to use element-wise differentiation in the input layer
        :param feature_size: embeddings size of binary features
        """
        super(CDCorefScorer, self).__init__()
        # length of embedding
        self.word_embed_dim = word_embeds.shape[1]
        self.char_embed_dim = char_embeds.shape[1]
        """LSTF的输入向量的长度"""
        self.char_hidden_dim = char_rep_size
        """LSTF的输出向量的长度"""

        # word embedding layer
        self.word_embed_layer = nn.Embedding(vocab_size, self.word_embed_dim)
        self.word_embed_layer.weight.data.copy_(torch.from_numpy(word_embeds))
        self.word_embed_layer.weight.requires_grad = False  # pre-trained word embeddings are fixed
        self.word_to_ix = word_to_ix

        # char embedding layer
        self.char_embed_layer = nn.Embedding(len(char_to_ix.keys()), self.word_embed_dim)
        self.char_embed_layer.weight.data.copy_(torch.from_numpy(char_embeds))
        self.char_embed_layer.weight.requires_grad = True
        self.char_to_ix = char_to_ix

        # char LSTM layer
        self.char_lstm_layer = nn.LSTM(input_size=self.char_embed_dim, hidden_size=self.char_hidden_dim, num_layers=1,
                                       bidirectional=False)

        # binary features for coreferring arguments/predicates
        self.coref_role_embeds = nn.Embedding(2, feature_size)

        # Linear layers
        self.input_dim = dims[0]
        self.hidden_dim_1 = dims[1]
        self.hidden_dim_2 = dims[2]
        self.out_dim = 1
        self.hidden_layer_1 = nn.Linear(self.input_dim, self.hidden_dim_1)
        self.hidden_layer_2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.out_layer = nn.Linear(self.hidden_dim_2, self.out_dim)

        # other flags
        self.use_mult = use_mult
        self.use_diff = use_diff
        self.model_type = 'CD_scorer'

    def forward(self, clusters_pair_tensor):
        '''
        The forward method - pass the input tensor through a feed-forward neural network
        :param clusters_pair_tensor: an input tensor consists of a concatenation between
        two mention representations, their element-wise multiplication and a vector of binary features
        (each feature embedded as 50 dimensional embeddings)
        :return: a predicted confidence score (between 0 to 1) of the mention pair to be in the
        same coreference chain (aka cluster).
        '''
        first_hidden = F.relu(self.hidden_layer_1(clusters_pair_tensor))
        second_hidden = F.relu(self.hidden_layer_2(first_hidden))
        out = F.sigmoid(self.out_layer(second_hidden))

        return out

    def init_char_hidden(self, device):
        '''
        initializes hidden states the character LSTM
        :param device: gpu/cpu Pytorch device
        :return: initialized hidden states (tensors)
        '''
        return (torch.randn((1, 1, self.char_hidden_dim), requires_grad=True).to(device),
                torch.randn((1, 1, self.char_hidden_dim), requires_grad=True).to(device))

    def get_char_embeds(self, seq, device):
        '''
        Runs a LSTM on a list of character embeddings and returns the last output state
        :param seq: a list of character embeddings
        :param device:  gpu/cpu Pytorch device
        :return: the LSTM's last output state
        '''
        char_hidden = self.init_char_hidden(device)
        input_char_seq = self.prepare_chars_seq(seq, device)
        char_embeds = self.char_embed_layer(input_char_seq).view(len(seq), 1, -1)
        char_lstm_out, char_hidden = self.char_lstm_layer(char_embeds, char_hidden)
        char_vec = char_lstm_out[-1]

        return char_vec

    def prepare_chars_seq(self, seq, device):
        '''
        Given a string represents a word or a phrase, this method converts the sequence
        to a list of character embeddings
        :param seq: a string represents a word or a phrase
        :param device: device:  gpu/cpu Pytorch device
        :return: a list of character embeddings
        '''
        idxs = []
        for w in seq:
            if w in self.char_to_ix:
                idxs.append(self.char_to_ix[w])
            else:
                lower_w = w.lower()
                if lower_w in self.char_to_ix:
                    idxs.append(self.char_to_ix[lower_w])
                else:
                    idxs.append(self.char_to_ix['<UNK>'])
                    print('can find char {}'.format(w))
        tensor = torch.tensor(idxs, dtype=torch.long).to(device)

        return tensor
