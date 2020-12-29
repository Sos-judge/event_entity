import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
from typing import Dict, List, Tuple, Union  # for type hinting

from src.all_models.models import CDCorefScorer
from src.all_models.model_utils import loadGloveWordEmbedding,loadGloveCharEmbeddings, load_one_hot_char_embeddings

word_embeds: np.ndarray = None
"""
A array with size(|V|, |w|).
|v| is the length of vocabulary.
|w| is the length of word embedding and is 300 by default beacause we use glove.6B.300d.txt by default.
The element word_embeds[i] is the word embedding of the i-th word in vocabulary. 
"""
word_to_ix = None

char_embeds = None
char_to_ix = None

'''
All functions in this script requires a configuration dictionary which contains flags and 
other attributes for configuring the experiments.
In this project, the configuration dictionaries are stored as JSON files (e.g. train_config.json)
and are loaded before the training/inference starts.
'''


def factory_load_embeddings(config_dict):
    '''
    Given a configuration dictionary, containing the paths to the embeddings files,
    this function loads the initial character embeddings and pre-trained word embeddings.
    and save it into global variables: ord_embeds, word_to_ix, char_embeds, char_to_ix

    :param config_dict: s configuration dictionary
    :return: no return, but set global variables: ord_embeds, word_to_ix, char_embeds,
     char_to_ix
    '''
    global word_embeds, word_to_ix, char_embeds, char_to_ix
    word_embeds, word_to_ix, char_embeds, char_to_ix = load_model_embeddings(config_dict)


def create_model(config_dict: Dict):
    '''
    Given a configuration dictionary, containing flags for configuring the current experiment,
    this function creates a model according to those flags and returns that model.

    :param config_dict: a configuration dictionary
    :return: an initialized model - src.all_models.CDCorefScorer object
    '''
    global word_embeds, word_to_ix, char_embeds, char_to_ix

    context_vector_size = 1024

    if config_dict["use_args_feats"]:
        mention_rep_size = context_vector_size + \
                            ((word_embeds.shape[1] + config_dict["char_rep_size"]) * 5)
    else:
        mention_rep_size = context_vector_size + word_embeds.shape[1] + config_dict["char_rep_size"]

    input_dim = mention_rep_size * 3

    if config_dict["use_binary_feats"]:
        input_dim += 4 * config_dict["feature_size"]

    second_dim = int(input_dim / 2)
    third_dim = second_dim
    model_dims = [input_dim, second_dim, third_dim]

    model = CDCorefScorer(word_embeds, word_to_ix, word_embeds.shape[0],
                          char_embedding=char_embeds, char_to_ix=char_to_ix,
                          char_rep_size=config_dict["char_rep_size"],
                          dims=model_dims,
                          use_mult=config_dict["use_mult"],
                          use_diff=config_dict["use_diff"],
                          feature_size=config_dict["feature_size"])

    return model


def create_optimizer(config_dict, model):
    '''
    Given a configuration dictionary, containing the string attribute "optimizer" that determines
    in which optimizer to use during the training.

    :param config_dict: a configuration dictionary
    :param model: an initialized CDCorefScorer object
    :return: Pytorch optimizer
    '''
    lr = config_dict["lr"]
    optimizer = None
    parameters = filter(lambda p: p.requires_grad,model.parameters())
    if config_dict["optimizer"] == 'adadelta':
        optimizer = optim.Adadelta(parameters, lr=lr,
                                   weight_decay=config_dict["weight_decay"])
    elif config_dict["optimizer"] == 'adam':
        optimizer = optim.Adam(parameters, lr=lr, weight_decay=config_dict["weight_decay"])
    elif config_dict["optimizer"] == 'sgd':
        optimizer = optim.SGD(parameters, lr=lr, momentum=config_dict["momentum"],
                              nesterov=True)

    assert (optimizer is not None), "Config error, check the optimizer field"

    return optimizer


def create_loss(config_dict):
    '''
    Given a configuration dictionary, containing the string attribute "loss" that determines
    in which loss function to use during the training.
    :param config_dict: a configuration dictionary
    :param model: an initialized CDCorefScorer object
    :return: Pytorch loss function

    '''
    loss_function = None

    if config_dict["loss"] == 'bce':
        loss_function = nn.BCELoss()

    assert (loss_function is not None), "Config error, check the loss field"

    return loss_function


def load_model_embeddings(config_dict: dict) -> tuple:
    '''
    Given a configuration dictionary like::

        {
            "glove_path":,
            "use_pretrained_char":,
            "char_pretrained_path":,
            "char_vocab_path":
        }

    * this function loads pre-trained word embeddings based on config_dict["glove_path"],
      and save the word embeddings into global variable **word_embeds**, **word_to_ix**.

    * If config_dict["use_pretrained_char"] is True, this function loads the initial
      character embeddings based on config_dict["char_pretrained_path"] and
      config_dict["char_vocab_path"], and save the character embeddings into global varable
      **char_embeds**, **char_to_ix**.

    :param config_dict: s configuration dictionary
    :returns: global variables: word_embeds, word_to_ix, char_embeds, char_to_ix.
     word_embeds and char_embeds is a ndarray. word_to_ix and char_to_ix is dict.

    '''
    logging.info('Loading word embeddings...')
    # load glove word embeddings
    word_vocab, word_embds = loadGloveWordEmbedding(config_dict["glove_path"])
    word_embeds = np.asarray(word_embds, dtype=np.float64)
    i = 0
    word_to_ix = {}
    for word in word_vocab:
        if word in word_to_ix:
           print("warning: word %s occurs multi times in word vocab.")
        word_to_ix[word] = i
        i += 1
    logging.info('Word embeddings have been loaded.')
    # load glove char embeddings
    if config_dict["use_pretrained_char"]:
        logging.info('Loading pre-trained char embeddings...')
        char_embeds, char_vocab = loadGloveCharEmbeddings(config_dict["char_pretrained_path"],
                                                          config_dict["char_vocab_path"])
        char_to_ix = {}
        for char in char_vocab:
            char_to_ix[char] = len(char_to_ix)
        # add special char " "
        char_to_ix[' '] = len(char_to_ix)
        space_vec = np.zeros((1, char_embeds.shape[1]))
        char_embeds = np.append(char_embeds, space_vec, axis=0)
        # add special char "<UNK>"
        char_to_ix['<UNK>'] = len(char_to_ix)
        unk_vec = np.random.rand(1, char_embeds.shape[1])
        char_embeds = np.append(char_embeds, unk_vec, axis=0)
        logging.info('Char embeddings have been loaded.')
    # load one-hot char embeddings
    else:
        logging.info('Loading one-hot char embeddings...')
        char_embeds, char_to_ix = load_one_hot_char_embeddings(config_dict["char_vocab_path"])

    return word_embeds, word_to_ix, char_embeds, char_to_ix