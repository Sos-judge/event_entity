import copy
import os
import sys
import json
import torch
import random
import logging
import itertools
import collections
import numpy as np
import _pickle as cPickle
from typing import Dict, List, Tuple, Union, Optional  # for type hinting

from src.all_models.bcubed_scorer import *
from scorer import *
from src.shared.eval_utils import *
from src.all_models.models import CDCorefScorer
from src.shared.classes import *
from src.shared.classes import Corpus, Topic, Document, Sentence, Mention, EventMention, EntityMention, Token, Srl_info, Cluster
# import matplotlib.pyplot as plt
# import spacy
# from spacy.lang.en import English

# for pack in os.listdir("src"):
#     sys.path.append(os.path.join("src", pack))
#
# sys.path.append("/src/shared/")


clusters_count = 1

analysis_pair_dict = {}


def get_topic(id):
    '''
    Extracts the topic id from the document ID.
    Note that this function doesn't extract the sub-topic ID (including the ecb/ecbplus notation)
    :param id: document id (string)
    :return: the topic id (string)
    '''
    return id.split('_')[0]


def merge_sub_topics_to_topics(test_set):
    '''
    Merges the test's sub-topics sub-topics to their topics (for experimental use).
    :param test_set: A Corpus object represents the test set
    :return: a dictionary contains the merged topics
    '''
    new_topics = {}
    topics_keys = test_set.topics.keys()
    for topic_id in topics_keys:
        topic = test_set.topics[topic_id]
        if get_topic(topic_id) not in new_topics:
            new_topics[get_topic(topic_id)] = Topic(get_topic(topic_id))
        new_topics[get_topic(topic_id)].docs.update(topic.docs)

    return new_topics


def load_predicted_topics(test_set: Corpus, config_dict: dict) -> dict:
    '''
    ecb语料库中是一个topic包含多个doc，即为真实的topic-doc对应关系。现在我抛弃这个对应关系，
    基于别的文档聚类算法，直接根据doc做文档聚类，得到预测的topic-doc对应关系。
    这个函数的作用就是把按照真实topic-doc对应关系组织的语料，根据预测的topic-doc对应关系重新排序组织。

    :param test_set: 原测试集，按照真实topic-doc对应关系组织document实例。
    :param config_dict: 字典对象，其中"predicted_topics_path"项是一个描述文件路径的字符串，文件保存
    了预测的topic-doc对应关系(即文档聚类结果)。
    :return:  新测试集，按照预测topic-doc对应关系组织document实例。
    '''
    print('\n按照外部文档聚类结果重新组织测试集：开始...', end='')
    # 1.把所有文档对象从原测试集中抽取出来（舍弃真实topic-doc对应关系）
    # 1.1.遍历得到文档对象的列表
    all_docs = []
    for topic in test_set.topics.values():
        all_docs.extend(topic.docs.values())
    '''
    all_docs = [Document对象, Document对象, ...]
    '''
    # 1.2.把列表转成字典
    all_doc_dict = {doc.doc_id: doc for doc in all_docs}
    '''
        all_docs = {'36_1ecb':Document对象, '36_2ecb':Document对象, ...}
    '''

    # 2.根据配置从文件中加载“预测的topic-doc对应关系”(文档聚类结果)
    with open(config_dict["predicted_topics_path"], 'rb') as f:
        predicted_topics = cPickle.load(f)
    ''' 
        predicted_topics是文档聚类结果
        predicted_topics = [
            ['45_6ecb', '45_8ecb'],  # 这是一个簇
            ['43_6ecb',43_4ecb]，  # 这是一个簇
            ...
        ]
    '''

    # 3.把抽取出来的文档对象按照预测的topic-doc对应关系重新组织
    new_topics = {}  # 新的测试集
    topic_counter = 1
    # 按照预测的topic-doc对应关系遍历，构建新的测试集
    for topic in predicted_topics:  # topic=['45_6ecb', '45_8ecb',...]
        # 在新的测试集中构建topic（但是topic是空的，下面还没有doc）
        topic_id = str(topic_counter)
        new_topics[topic_id] = src.shared.classes.Topic(topic_id)
        print('\nTopic', topic_id, ':', end='')
        # 把原测试集中的doc对象挪到新测试集的topic下面
        for doc_name in topic:
            print(doc_name, end='')
            if doc_name in all_doc_dict:
                new_topics[topic_id].docs[doc_name] = all_doc_dict[doc_name]
        topic_counter += 1
    print('共有', len(new_topics), '个topic')
    print('按照外部文档聚类结果重新组织测试集：结束...')
    return new_topics


def topic_to_mention_list(topic: Topic, is_gold: bool) -> Tuple[List[EventMention], List[EntityMention]]:
    """
    抽取topic中的gold mention或predicted mention(取决于is_gold参数),并组成列表返回。

    :param topic: a Topic object
    :param is_gold: a flag that denotes whether to extract gold mention or predicted mentions
    :return: (gold event mention list, gold entity mention list)或(predicted event mention list, predicted
     entity mention list)
    """
    event_mentions = []
    entity_mentions = []
    for doc_id, doc in topic.docs.items():  # 遍历每个doc
        for sent_id, sent in doc.sentences.items():  # 遍历每个sent(句子)
            if is_gold:  # 抽取gold_mention
                event_mentions.extend(sent.gold_event_mentions)
                entity_mentions.extend(sent.gold_entity_mentions)
            else:  # 抽取predicted_mention
                event_mentions.extend(sent.pred_event_mentions)
                entity_mentions.extend(sent.pred_entity_mentions)

    return event_mentions, entity_mentions


def load_entity_wd_clusters(config_dict: Dict) -> Dict[str, Dict[str, any]]:
    '''
    Loads from a file the within-document (WD) entity coreference clusters predicted by an external WD entity coreference
    model/tool and ordered those clusters in a dictionary according to their documents.其实就是根据配置读取外部WD实体共指
    的结果，然后按照“文档id-句子id”的二级顺序结构排个序。

    :param config_dict: a configuration dictionary that contains a path to a file stores the
     within-document (WD) entity coreference clusters predicted by an external WD entity coreference
     system.一个字典对象，其中"wd_entity_coref_file"项的值是一个地址，指向“外部WD实体共指
     的结果”
    :return: 按照行文顺序（文章序号+句子序号）排序后的“外部WD实体共指的结果”。数据结构为{ 'doc_id文档序号': {'sent_id句子序号': [共指信息]}   }
    '''

    doc_to_entity_mentions = {}

    # 读取json文件
    with open(config_dict["wd_entity_coref_file"], 'r') as js_file:
        # 把json文件反序列化为变量
        js_mentions = json.load(js_file)
        '''
        js_mentions =[
            {'doc_id': '???.xml', 'sent_id': ?, 'tokens_numbers':[?,?],....},
            {'doc_id': '???.xml', 'sent_id': ?, 'tokens_numbers':[?,?],....},
            ...,
            {'doc_id': '???.xml', 'sent_id': ?, 'tokens_numbers':[?,?],....}
        ]
        '''

    # load all entity mentions in the json
    for js_mention in js_mentions:
        # 1.按照[doc_id][sent_id]的层级，构建doc_to_entity_mentions的二级顺序结构
        # 1.1 添加一级元素doc_id（不重复添加）
        doc_id = js_mention["doc_id"].replace('.xml', '')
        if doc_id not in doc_to_entity_mentions:
            doc_to_entity_mentions[doc_id] = {}
        # 1.2 添加二级元素sent_id（不重复添加）
        sent_id = js_mention["sent_id"]
        if sent_id not in doc_to_entity_mentions[doc_id]:
            doc_to_entity_mentions[doc_id][sent_id] = []
        # 2.doc_to_entity_mentions字典中二级顺序结构构建完成，接下来就是把json文件中的数据转移过来
        tokens_numbers = js_mention["tokens_numbers"]
        mention_str = js_mention["tokens_str"]
        try:
            coref_chain = js_mention["coref_chain"]
        except:
            continue
        # 3.填充特征
        doc_to_entity_mentions[doc_id][sent_id].append((
            doc_id, sent_id, tokens_numbers, mention_str, coref_chain
        ))
    return doc_to_entity_mentions


def init_entity_wd_clusters(entity_mentions, doc_to_entity_mentions):
    '''
    Matches entity mentions with their predicted within-document coreference clusters
    produced by an external within-document entity coreference system and forms the initial
    within-document entity coreference clusters.

    :param entity_mentions: gold entity mentions (currently doesn't support
     predicted entity mentions).
    :param doc_to_entity_mentions: a dictionary contains a mapping of a documents to
    their predicted entity clusters.
    :return: a list of Cluster objects contains the initial within-document entity coreference
    clusters
    '''

    doc_to_clusters = {}
    all_entity_clusters = {}

    for entity in entity_mentions:
        doc_id = entity.doc_id
        sent_id = entity.sent_id
        is_entity_found = False
        found_entity = None
        if doc_id in doc_to_entity_mentions and sent_id in doc_to_entity_mentions[doc_id]:
            predicted_entity_mentions = doc_to_entity_mentions[doc_id][sent_id]

            for pred_entity in predicted_entity_mentions:
                pred_start = pred_entity[2][0]
                pred_end = pred_entity[2][-1]
                pred_str = pred_entity[3]
                if have_string_match(entity,pred_str ,pred_start, pred_end):
                    is_entity_found = True
                    found_entity = pred_entity
                    break

        if is_entity_found:
            if doc_id not in doc_to_clusters:
                doc_to_clusters[doc_id] = {}
            pred_coref_chain = found_entity[4]
            if pred_coref_chain not in doc_to_clusters[doc_id]:
                doc_to_clusters[doc_id][pred_coref_chain] = src.shared.classes.Cluster(is_event=False)
            doc_to_clusters[doc_id][pred_coref_chain].mentions[entity.mention_id] = entity
        else:
            doc_id = entity.doc_id
            if doc_id not in all_entity_clusters:
                all_entity_clusters[doc_id] = []

            singleton = Cluster(is_event=False)
            singleton.mentions[entity.mention_id] = entity
            all_entity_clusters[doc_id].append(singleton)

    count_matched_clusters = 0

    for doc_id, doc_coref_chains in doc_to_clusters.items():
        for coref_chain, cluster in doc_coref_chains.items():
            if doc_id not in all_entity_clusters:
                all_entity_clusters[doc_id] = []
            all_entity_clusters[doc_id].append(cluster)
            if len(cluster.mentions.values()) > 1:
                count_matched_clusters += 1

    logging.info('Matched non-singleton clusters {}'.format(count_matched_clusters))

    return all_entity_clusters


mention_list_to_external_wd_cluster_dict = init_entity_wd_clusters


def mention_list_to_external_wd_cluster_list(mention_list: list, external_wd_coref_info,
                                             is_event: bool) -> list:
    if is_event is True:
        print("mention_list_to_external_wd_cluster_list 暂不支持is_event为True")
        return []
    mention_dict = mention_list_to_external_wd_cluster_dict(mention_list, external_wd_coref_info)
    cluster_list = []
    for doc_id, clusters in mention_dict.items():
        cluster_list.extend(clusters)
    return cluster_list


def have_string_match(mention, pred_str, pred_start, pred_end):
    '''
    Checks whether a mention has a match (strict or relaxed) with a predicted mention.
    Used when initializing within-document (WD) entity coreference clusters with the output of
    an external (WD) coreference system.
    :param mention: an EntityMention object
    :param pred_str: predicted mention's text (string)
    :param pred_start: predicted mention's start offset
    :param pred_end: predicted mention's end offset
    :return: True if a match has been found and false otherwise.
    '''
    if mention.mention_str == pred_str and mention.start_offset == pred_start:
        return True
    if mention.mention_str == pred_str:
        return True
    if mention.start_offset >= pred_start and mention.end_offset <= pred_end:
        return True
    if pred_start >= mention.start_offset and pred_end <= mention.end_offset:
        return True

    return False


def init_wd(mention_list, is_event):
    """Mention list -> Cluster dict (singleton cluster)

    * mention_list = [Mention, ...]
    * cluster_dict[doc id] = [Cluster, ...] (this Cluster is singleton cluster)

    :param mention_list:  a set of Mention objects (either EventMention or EntityMention)
    :param is_event: whether the mentions are event or entity mentions.
    :return: a dictionary contains initial singleton clusters, ordered by the mention's
     document ID.
    """
    cluster_dict = {}
    for mention in mention_list:
        mention_doc_id = mention.doc_id
        if mention_doc_id not in cluster_dict:
            cluster_dict[mention_doc_id] = []
        cluster = Cluster(is_event=is_event)
        cluster.mentions[mention.mention_id] = mention
        cluster_dict[mention_doc_id].append(cluster)
    return cluster_dict


mention_list_to_singleton_cluster_dict = init_wd


def init_cd(mention_list, is_event):
    """Mention list -> Cluster list (singleton cluster)

    * mention_list = [Mention, ...]
    * cluster_list = [Cluster, ...] (this Cluster is singleton cluster)

    :param mention_list:  Mention list (either event or entity mention)
    :param is_event: whether the mentions are event or entity mentions.
    :return: Cluster list (singleton clusters).
    """
    cluster_list = []
    for mention in mention_list:
        cluster = Cluster(is_event=is_event)
        cluster.mentions[mention.mention_id] = mention
        cluster_list.append(cluster)
    return cluster_list


mention_list_to_singleton_cluster_list = init_cd


def loadGloveCharEmbeddings(embed_path, vocab_path):
    '''
    load embeddings from a binary file and a file contains the vocabulary.
    :param embed_path: path to the embeddings' binary file
    :param vocab_path: path to the vocabulary file
    :return: word_embeds - a numpy array containing the word vectors, vocab - a list containing the
    vocabulary.
    '''
    with open(embed_path,'rb') as f:
        word_embeds = np.load(f)

    vocab = []
    for line in open(vocab_path, 'r'):
        vocab.append(line.strip())

    return word_embeds, vocab


def load_one_hot_char_embeddings(char_vocab_path):
    '''
    Loads character vocabulary and creates one hot embedding to each character which later
    can be used to initialize the character embeddings (experimental)
    :param char_vocab_path: a path to the vocabulary file
    :return: char_embeds - a numpy array containing the char vectors, vocab - a list containing the
    vocabulary.
    '''
    vocab = []
    for line in open(char_vocab_path, 'r'):
        vocab.append(line.strip())

    char_to_ix = {}
    for char in vocab:
        char_to_ix[char] = len(char_to_ix)

    char_to_ix[' '] = len(char_to_ix)
    char_to_ix['<UNK>'] = len(char_to_ix)

    char_embeds = np.eye(len(char_to_ix))

    return char_embeds, char_to_ix


def is_stop(w):
    '''
    Checks whether w is a stop word according to a small list of stop words.
    :param w: a word (string)
    :return: True is w is a stop word and false otherwise.
    '''
    return w.lower() in ['a', 'an', 'the', 'in', 'at', 'on','for','very']


def clean_word(word):
    '''
    Removes apostrophes(' or 's or ") before look for a word in the word embeddings vocabulary.
    :param word: a word (string)
    :return: the word (string) after removing the apostrophes.
    '''
    word = word.replace("'s",'').replace("'",'').replace('"','')
    return word


def get_char_embed(word, model, device):
    '''
    Runs a character LSTM over a word/phrase and returns the LSTM's output vector
    :param word: a word/phrase (string)
    :param model: CDCorefScorer object
    :param device: Pytorch device (gpu/cpu)
    :return:  the character-LSTM's last output vector
    '''
    char_vec = model.get_char_embeds(word, device)

    return char_vec


def find_word_embed(word: str, model: CDCorefScorer, device: torch.cuda.device) -> torch.Tensor:
    """
    This function get the embedding vector of *word* from the word embedding layer in *model*.
    In the setting of Barhom2019, the word embedding layer is Glove300d.

    :param word: A word.
    :param model: A CDCorefScorer object which has a word embedding layer.
    :param device: Pytorch device
    :return: The embedding vector of *word*
    """
    word_to_ix = model.word_to_ix
    word = clean_word(word)
    # get word index
    if word in word_to_ix:
        word_ix = [word_to_ix[word]]
    elif word.lower() in word_to_ix:
        word_ix = [word_to_ix[word.lower()]]
    else:
        word_ix = [word_to_ix['unk']]
    # get word embedding
    word_tensor = model.word_embed_layer(torch.tensor(word_ix, dtype=torch.long).to(device))
    #
    return word_tensor


def find_mention_cluster(mention_id, clusters):
    '''
    Given a mention ID, the function fetches its current predicted cluster.
    :param mention_id: mention ID
    :param clusters: current clusters, should be of the same type (event/entity) as the mention.
    :return: the mention's current predicted cluster
    '''
    for cluster in clusters:
        if mention_id in cluster.mentions:
            return cluster
    raise ValueError('Can not find mention cluster!')


def is_system_coref(mention_id_1, mention_id_2, clusters):
    '''
    Checks whether two mentions are in the same predicted (system) clusters in the current
    clustering configuration.
    :param mention_id_1: first mention ID
    :param mention_id_2: second menton ID
    :param clusters: current clustering configuration (should be of the same type as the mentions,
    e.g. if mention_1 and mention_2 are event mentions, so clusters should be the current event
    clusters)
    :return: True if both mentions belong to the same cluster and false otherwise
    '''
    cluster_1 = find_mention_cluster(mention_id_1, clusters)
    cluster_2 = find_mention_cluster(mention_id_2, clusters)

    if cluster_1 == cluster_2:
        return True
    return False


def create_args_features_vec(mention_1, mention_2, entity_clusters, device, model):
    '''
    Creates a vector for four binary features (one for each role - Arg0/Arg1/location/time)
    indicate whether two event mentions share a coreferrential argument in the same role.
    :param mention_1: EventMention object
    :param mention_2: EventMention object
    :param entity_clusters: current entity clusters
    :param device: Pytorch device (cpu/gpu)
    :param model: CDCorefScorer object
    :return: a vector for four binary features embedded as a tensor of size (1,200),
    each feature embedded as 50 dimensional embedding.

    '''
    coref_a0 = 0
    coref_a1 = 0
    coref_loc = 0
    coref_tmp = 0

    if coref_a0 == 0 and mention_1.arg0 is not None and mention_2.arg0 is not None:
        if is_system_coref(mention_1.arg0[1], mention_2.arg0[1],entity_clusters):
            coref_a0 = 1
    if coref_a1 == 0 and mention_1.arg1 is not None and mention_2.arg1 is not None:
        if is_system_coref(mention_1.arg1[1], mention_2.arg1[1],entity_clusters):
            coref_a1 = 1
    if coref_loc == 0 and mention_1.amloc is not None and mention_2.amloc is not None:
        if is_system_coref(mention_1.amloc[1], mention_2.amloc[1],entity_clusters):
            coref_loc = 1
    if coref_tmp == 0 and mention_1.amtmp is not None and mention_2.amtmp is not None:
        if is_system_coref(mention_1.amtmp[1], mention_2.amtmp[1],entity_clusters):
            coref_tmp = 1

    arg0_tensor = model.coref_role_embeds(torch.tensor(coref_a0,
                                                       dtype=torch.long).to(device)).view(1,-1)
    arg1_tensor = model.coref_role_embeds(torch.tensor(coref_a1,
                                                       dtype=torch.long).to(device)).view(1,-1)
    amloc_tensor = model.coref_role_embeds(torch.tensor(coref_loc,
                                                        dtype=torch.long).to(device)).view(1,-1)
    amtmp_tensor = model.coref_role_embeds(torch.tensor(coref_tmp,
                                                        dtype=torch.long).to(device)).view(1,-1)

    args_features_tensor = torch.cat([arg0_tensor,arg1_tensor, amloc_tensor,amtmp_tensor],1)

    return args_features_tensor


def create_predicates_features_vec(mention_1, mention_2, event_clusters, device, model):
    '''
    Creates a vector for four binary features (one for each role - Arg0/Arg1/location/time)
    indicate whether two entity mentions share a coreferrential predicate in the same role.
    :param mention_1: EntityMention object
    :param mention_2: EntityMention object
    :param event_clusters: current entity clusters
    :param device: Pytorch device (cpu/gpu)
    :param model: CDCorefScorer object
    :return: a vector for four binary features embedded as a tensor of size (1,200),
    each feature embedded as 50 dimensional embedding.

    '''
    coref_pred_a0 = 0
    coref_pred_a1 = 0
    coref_pred_loc = 0
    coref_pred_tmp = 0

    predicates_dict_1 = mention_1.predicates
    predicates_dict_2 = mention_2.predicates
    for predicate_id_1, rel_1 in predicates_dict_1.items():
        for predicate_id_2, rel_2 in predicates_dict_2.items():
            if coref_pred_a0 == 0 and rel_1 == 'A0' and rel_2 == 'A0':
                if is_system_coref(predicate_id_1[1], predicate_id_2[1], event_clusters):
                    coref_pred_a0 = 1
            if coref_pred_a1 == 0 and rel_1 == 'A1' and rel_2 == 'A1':
                if is_system_coref(predicate_id_1[1], predicate_id_2[1], event_clusters):
                    coref_pred_a1 = 1
            if coref_pred_loc == 0 and rel_1 == 'AM-LOC' and rel_2 == 'AM-LOC':
                if is_system_coref(predicate_id_1[1], predicate_id_2[1], event_clusters):
                    coref_pred_loc = 1
            if coref_pred_tmp == 0 and rel_1 == 'AM-TMP' and rel_2 == 'AM-TMP':
                if is_system_coref(predicate_id_1[1], predicate_id_2[1], event_clusters):
                    coref_pred_tmp = 1

    arg0_tensor = model.coref_role_embeds(torch.tensor(coref_pred_a0,
                                                       dtype=torch.long).to(device)).view(1,-1)
    arg1_tensor = model.coref_role_embeds(torch.tensor(coref_pred_a1,
                                                       dtype=torch.long).to(device)).view(1,-1)
    amloc_tensor = model.coref_role_embeds(torch.tensor(coref_pred_loc,
                                                        dtype=torch.long).to(device)).view(1,-1)
    amtmp_tensor = model.coref_role_embeds(torch.tensor(coref_pred_tmp,
                                                        dtype=torch.long).to(device)).view(1,-1)

    predicates_features_tensor = torch.cat([arg0_tensor, arg1_tensor, amloc_tensor, amtmp_tensor],1)

    return predicates_features_tensor


def float_to_tensor(float_num, device):
    '''
    Convert a floating point number to a tensor
    :param float_num: a floating point number
    :param device: Pytorch device (cpu/gpu)
    :return: a tensor
    '''
    float_tensor = torch.tensor([float(float_num)], requires_grad=False).to(device).view(1, -1)

    return float_tensor


def calc_q(cluster_1, cluster_2):
    '''
    Calculates the quality of merging two clusters, denotes by the proportion between
    the number gold coreferrential mention pairwise links (between the two clusters) and all the
    pairwise links.

    :param cluster_1: first cluster
    :param cluster_2: second cluster
    :return: the quality of merge (a number between 0 to 1)
    '''
    true_pairs = 0
    false_pairs = 0
    for mention_c1 in cluster_1.mentions.values():
        for mention_c2 in cluster_2.mentions.values():
            if mention_c1.gold_tag == mention_c2.gold_tag:
                true_pairs += 1
            else:
                false_pairs += 1

    return true_pairs/float(true_pairs + false_pairs)


def loadGloveWordEmbedding(glove_filename: str) -> Tuple[List, List]:
    '''
    Loads Glove word vectors.

    :param glove_filename: Glove file
    :return: vocab - list contains the vocabulary ,embd - list of word vectors
    '''
    #
    vocab = []
    embd = []
    #
    with open(glove_filename, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            row = line.strip().split(' ')
            if len(row) > 1:
                if row[0] != '':
                    vocab.append(row[0])
                    embd.append(row[1:])
                    if len(row[1:]) != 300:
                        logging.info("warning: len of embedding of word %s is not 300." % (vocab[-1]))
    #
    return vocab, embd


def get_sub_topics(doc_id):
    '''
    Extracts the sub-topic id from the document ID.
    :param doc_id: document id (string)
    :return: the sub-topic id (string)
    '''
    topic = doc_id.split('_')[0]
    if 'ecbplus' in doc_id:
        category = 'ecbplus'
    else:
        category = 'ecb'
    return '{}_{}'.format(topic, category)


def separate_clusters_to_sub_topics(clusters, is_event):
    '''
    Removes spurious cross sub-topics coreference link (used for experiments in Yang setup).
    :param clusters: a list of Cluster objects
    :param is_event: Clusters' type (event/entity)
    :return: new list of clusters, after spurious cross sub-topics coreference link were removed.
    '''
    new_clusters = []
    for cluster in clusters:
        sub_topics_to_clusters = {}
        for mention in cluster.mentions.values():
            mention_sub_topic = get_sub_topics(mention.doc_id)
            if mention_sub_topic not in sub_topics_to_clusters:
                sub_topics_to_clusters[mention_sub_topic] = []
            sub_topics_to_clusters[mention_sub_topic].append(mention)
        for sub_topic, mention_list in sub_topics_to_clusters.items():
            new_cluster = Cluster(is_event)
            for mention in mention_list:
                new_cluster.mentions[mention.mention_id] = mention
            new_clusters.append(new_cluster)

    return new_clusters


def set_coref_chain_to_mentions(clusters, is_event, is_gold, intersect_with_gold,):
    '''
    Sets the predicted cluster id to all mentions in the cluster
    :param clusters: predicted clusters (a list of Corpus objects)
    :param is_event: True, if clusters are event clusters, False otherwise - currently unused.
    :param is_gold: True, if the function sets gold mentions and false otherwise
     (it sets predicted mentions) - currently unused.
    :param intersect_with_gold: True, if the function sets predicted mentions that were matched
    with gold mentions (used in setting that requires to match predicted mentions with gold
    mentions - as in Yang's setting) , and false otherwise - currently unused.
    :param remove_singletons: True if the function ignores singleton clusters (as in Yang's setting)
    '''
    global clusters_count
    for cluster in clusters:
        cluster.cluster_id = clusters_count
        for mention in cluster.mentions.values():
            mention.cd_coref_chain = clusters_count
        clusters_count += 1


def save_check_point(model, fname):
    '''
    Saves Pytorch model to a file
    :param model: Pytorch model
    :param fname: output filename
    '''
    torch.save(model, fname)


def load_check_point(fname, config_dict):
    '''
    Loads Pytorch model from a file
    :param fname: model's filename
    :return:Pytorch model
    '''
    if config_dict["gpu_num"] != -1: # gpu_num =-1时使用cpu 原先写的是use_cuda 但是配置文件里没有该词条
        return torch.load(fname)
    else:
        return torch.load(fname, map_location=torch.device('cpu'))


def create_gold_clusters(mentions):
    """Mention list -> Mention dict (by doc and gold WD mention cluster)

    Given a mention objs list, the mention obj has gold WD coref info(WD mention cluster info),
    this function rearrange those mention objs as a dict based on the gold WD coref info, such as
    wd_clusters[doc_id][instance_id(cluster_id)]=[a list of mention obj that in the same WD mention cluster].

    :param mentions: mention obj list
    :return: mention obj dict that is arranged by doc and gold WD mention cluster.
    """
    wd_clusters = {}
    for mention in mentions:
        mention_doc_id = mention.doc_id
        if mention_doc_id not in wd_clusters:
            wd_clusters[mention_doc_id] = {}
        mention_gold_tag = mention.gold_tag
        if mention_gold_tag not in wd_clusters[mention_doc_id]:
            wd_clusters[mention_doc_id][mention_gold_tag] = []
        wd_clusters[mention_doc_id][mention_gold_tag].append(mention)

    return wd_clusters


def create_gold_wd_clusters_organized_by_doc(mention_list: List[Mention], is_event: bool) -> Dict[str, List[Cluster]]:
    """
    This function:
        1. extract gold wd coref clusters from mention list.
        2. The extracted clusters make up a dict by doc_id. Return the dict.

    :param mention_list: Event mentions list or entity metions list. Note that Mention obj has doc and gold WD coref info.
    :param is_event: The mention in *mention_list* is event mention or entity mention.
    :return: Dict{doc_id: [Cluster_obj_in_cur_doc, ...], ...}.(this Cluster is gold WD coref cluster)
    """

    # Mention list -> Mention dict
    """
    mention_list = [Mention, ...]  
    mention_dict[doc id][cluster id] = [Mention, ...]. (this Cluster is gold WD coref cluster)
    """
    mention_dict = create_gold_clusters(mention_list)

    # Mention dict -> Cluster dict
    """
    mention_dict[doc id][cluster id] = [Mention, ...]. (this Cluster is gold WD coref cluster)
    cluster_dict[doc id] = [Cluster, ...]. (this Cluster is gold WD coref cluster)
    """
    cluster_dict = {}
    for doc_id, gold_chain_in_doc in mention_dict.items():
        for gold_chain_id, gold_chain in gold_chain_in_doc.items():
            cluster = Cluster(is_event)
            for mention in gold_chain:
                cluster.mentions[mention.mention_id] = mention
            if doc_id not in cluster_dict:
                cluster_dict[doc_id] = []
            cluster_dict[doc_id].append(cluster)

    return cluster_dict


mention_list_to_gold_wd_cluster_dict = create_gold_wd_clusters_organized_by_doc


def mention_list_to_gold_wd_cluster_list(mention_list: List[Mention], is_event: bool):
    # get cluster dict
    cluster_dict = mention_list_to_gold_wd_cluster_dict(mention_list, is_event)
    # transfer cluster dict into cluster list
    cluster_list = []
    for cur_doc_id, cur_doc_cluster_list in cluster_dict.items():
        cluster_list.extend(cur_doc_cluster_list)
    #
    return cluster_list

def write_event_coref_results(corpus, out_dir, config_dict):
    '''
    Writes to a file (in a CoNLL format) the predicted event clusters (for evaluation).
    :param corpus: A Corpus object
    :param out_dir: output directory
    :param config_dict: configuration dictionary
    '''
    if not config_dict["test_use_gold_mentions"]:
        out_file = os.path.join(out_dir, 'CD_test_event_span_based.response_conll')
        src.shared.eval_utils.write_span_based_cd_coref_clusters(corpus, out_file, is_event=True, is_gold=False,
                                           use_gold_mentions=config_dict["test_use_gold_mentions"])
    else:
        out_file = os.path.join(out_dir, 'CD_test_event_mention_based.response_conll')
        src.shared.eval_utils.write_mention_based_cd_clusters(corpus, is_event=True, is_gold=False, out_file=out_file)

        out_file = os.path.join(out_dir, 'WD_test_event_mention_based.response_conll')
        src.shared.eval_utils.write_mention_based_wd_clusters(corpus, is_event=True, is_gold=False, out_file=out_file)


def write_entity_coref_results(corpus, out_dir,config_dict):
    '''
    Writes to a file (in a CoNLL format) the predicted entity clusters (for evaluation).
    :param corpus: A Corpus object
    :param out_dir: output directory
    :param config_dict: configuration dictionary
    '''
    if not config_dict["test_use_gold_mentions"]:
        out_file = os.path.join(out_dir, 'CD_test_entity_span_based.response_conll')
        src.shared.eval_utils.write_span_based_cd_coref_clusters(corpus, out_file, is_event=False, is_gold=False,
                                           use_gold_mentions=config_dict["test_use_gold_mentions"])
    else:
        out_file = os.path.join(out_dir, 'CD_test_entity_mention_based.response_conll')
        src.shared.eval_utils.write_mention_based_cd_clusters(corpus, is_event=False, is_gold=False, out_file=out_file)

        out_file = os.path.join(out_dir, 'WD_test_entity_mention_based.response_conll')
        src.shared.eval_utils.write_mention_based_wd_clusters(corpus, is_event=False, is_gold=False, out_file=out_file)


def create_event_cluster_bow_lexical_vec(event_cluster: Cluster, model: CDCorefScorer,
                                         device: torch.cuda.device, use_char_embeds: bool, requires_grad: bool):
    """
    Creates the semantically-dependent vector of an entity cluster (average of mention's span vectors in the cluster)
    计算簇向量。

    - 簇向量 = 簇内指称向量的平均值。
    - If *use_char_embeds* is true, 指称向量s(m) = (词级指称向量(m);字级指称向量(m));
    - If *use_char_embeds* is false, 指称向量s(m) = (词级指称向量(m));
    - 词级嵌入(m) = 指称内head word的glove词向量的平均值。
    - 字级嵌入(m) = 指称内head word的glove字向量的平均值。

    :param event_cluster: entity cluster
    :param model: CDCorefScorer model
    :param device: Pytorch device (gpu/cpu)
    :param use_char_embeds: whether to use character embeddings
    :param requires_grad: whether the tensors require gradients (True for
        training time and False for inference time)
    :return: semantically-dependent vector of a specific entity cluster
        (average of mention's span vectors in the cluster)
    """
    # init cluster_vec
    if use_char_embeds:
        bow_vec = torch.zeros(
            model.word_embed_dim + model.char_hidden_dim,
            requires_grad=requires_grad
        ).to(device).view(1, -1)
    else:
        bow_vec = torch.zeros(
            model.word_embed_dim,
            requires_grad=requires_grad
        ).to(device).view(1, -1)
    # set cluster_vec
    for event_mention in event_cluster.mentions.values():
        # 1. get word level embedding of cur event mention. 词级指称向量 = head的词向量
        head = event_mention.mention_head
        words_vec = find_word_embed(head, model, device)
        # 2. get char level embedding of cur entity mention. 字级指称向量
        if use_char_embeds:
            chars_vec = get_char_embed(head, model, device)
            if not requires_grad:
                chars_vec = chars_vec.detach()
        # 3. get embedding of cur entity mention. 指称向量 = (词级指称向量;字级指称向量)
        if use_char_embeds:
            # s(m) = (词级嵌入(m);字级嵌入(m))
            mention_vec = torch.cat([words_vec, chars_vec], 1)
        else:
            # s(m) = (词级嵌入(m))
            mention_vec = words_vec
        # 4. get embedding of cur entity cluster. 簇向量 = 簇内指称向量的平均值
        bow_vec += mention_vec
    return bow_vec / len(event_cluster.mentions.keys())


def create_entity_cluster_bow_lexical_vec(entity_cluster: Cluster, model: CDCorefScorer,
                                          device: torch.cuda.device, use_char_embeds: bool, requires_grad: bool):
    """
    Creates the semantically-dependent vector of an entity cluster (average of mention's span vectors in the cluster)
    计算簇向量。

    - 簇向量 = 簇内指称向量的平均值。
    - If *use_char_embeds* is true, 指称向量s(m) = (词级指称向量(m);字级指称向量(m));
    - If *use_char_embeds* is false, 指称向量s(m) = (词级指称向量(m));
    - 词级指称(m) = 指称内各词的glove词向量的平均值。
    - 字级嵌入(m) = ?

    :param entity_cluster: entity cluster
    :param model: CDCorefScorer model
    :param device: Pytorch device (gpu/cpu)
    :param use_char_embeds: whether to use character embeddings
    :param requires_grad: whether the tensors require gradients (True for
        training time and False for inference time)
    :return: semantically-dependent vector of a specific entity cluster
        (average of mention's span vectors in the cluster)
    """
    # init cluster_vec
    if use_char_embeds:
        cluster_vec = torch.zeros(
            model.word_embed_dim + model.char_hidden_dim,
            requires_grad=requires_grad
        ).to(device).view(1, -1)
    else:
        cluster_vec = torch.zeros(
            model.word_embed_dim,
            requires_grad=requires_grad
        ).to(device).view(1, -1)
    # set cluster_vec
    for entity_mention in entity_cluster.mentions.values():
        # 1. get word level embedding of cur entity mention. 词级指称向量 = 指称中每个词的词向量的平均
        if 1:
            # 1.1 init words_vec
            words_vec = torch.zeros(model.word_embed_dim, requires_grad=requires_grad).to(device).view(1, -1)
            """word level embedding of cur entity mention."""
            # 1.2 calc words_vec: 实体指称的词级嵌入等于指称内每个词的嵌入取平均
            words_vec_list: List[torch.Tensor] = [find_word_embed(word, model, device)
                                                  for word in entity_mention.get_tokens()
                                                  if not is_stop(word)]
            """GloVe embedding of each word in cur entity mention make up a list."""
            for word_vec in words_vec_list:
                words_vec += word_vec
            words_vec /= len(entity_mention.get_tokens())
        # 2. get char level embedding of cur entity mention. 字级指称向量 = 指称的字符串的字向量
        if use_char_embeds:
            chars_vec = get_char_embed(entity_mention.mention_str, model, device)
            """word level embedding of cur entity mention."""
            if not requires_grad:
                chars_vec = chars_vec.detach()
        # 3. get embedding of cur entity mention. 指称向量 = (词级指称向量;字级指称向量)
        if use_char_embeds:
            # s(m) = (词级嵌入(m);字级嵌入(m))
            mention_vec = torch.cat([words_vec, chars_vec], 1)
        else:
            # s(m) = (词级嵌入(m))
            mention_vec = words_vec
        # 4. get embedding of cur entity cluster. 簇向量 = 簇内指称向量的平均值
        cluster_vec += mention_vec
    return cluster_vec / len(entity_cluster.mentions.keys())


def find_mention_cluster_vec(mention_id: str, clusters: Cluster) -> torch.Tensor:
    """
    Fetches a semantically-dependent vector of a mention's cluster

    Given a mention id, this function:
    1. find out which cluster this mention belong to.
    2. return the semantically-dependent vector of this cluster, that is the_cluster.lex_vec.

    :param mention_id: mention ID.
    :param clusters: list of Cluster objects
    :return: semantically-dependent vector of a mention's cluster.
        Pytorch tensor with size (1, 350).
    """
    for cluster in clusters:
        if mention_id in cluster.mentions:
            return cluster.lex_vec.detach()


def create_event_cluster_bow_arg_vec(event_cluster: List[Cluster], entity_clusters: List[Cluster],
                                     model: CDCorefScorer, device: torch.cuda.device) -> None:
    """
    根据实体簇，更新事件簇中每个事件指称的语义依存向量（arg0，arg1，time， loc）。

    Creates the semantically-dependent vectors (of all roles) for all mentions
    in a specific event cluster.

    :param event_cluster: a Cluster object which contains EventMention objects.
    :param entity_clusters: current predicted entity clusters (a list)
    :param model: CDCorefScorer object
    :param device: Pytorch device
    :return: No return.
        But for each event mention in *event_cluster*, the_event_mention.arg0_vec/arg1_vec/time_vec/loc_vec are setted.
    """
    for event_mention in event_cluster.mentions.values():
        event_mention.arg0_vec = torch.zeros( model.word_embed_dim + model.char_hidden_dim, requires_grad=False ).to(device).view(1, -1)
        event_mention.arg1_vec = torch.zeros( model.word_embed_dim + model.char_hidden_dim, requires_grad=False ).to(device).view(1, -1)
        event_mention.time_vec = torch.zeros(model.word_embed_dim + model.char_hidden_dim, requires_grad=False).to(device).view(1, -1)
        event_mention.loc_vec = torch.zeros(model.word_embed_dim + model.char_hidden_dim, requires_grad=False).to(device).view(1, -1)
        if event_mention.arg0 is not None:
            event_mention.arg0_vec = find_mention_cluster_vec(event_mention.arg0[1], entity_clusters).to(device)
        if event_mention.arg1 is not None:
            event_mention.arg1_vec = find_mention_cluster_vec(event_mention.arg1[1], entity_clusters).to(device)
        if event_mention.amtmp is not None:
            event_mention.time_vec = find_mention_cluster_vec(event_mention.amtmp[1], entity_clusters).to(device)
        if event_mention.amloc is not None:
            event_mention.loc_vec = find_mention_cluster_vec(event_mention.amloc[1], entity_clusters).to(device)


def create_entity_cluster_bow_predicate_vec(entity_cluster: List[Cluster], event_clusters: List[Cluster],
                                            model: CDCorefScorer, device: torch.cuda.device):
    """
    根据事件簇，更新实体簇中每个实体指称的的语义依存向量（arg0，arg1，time， loc）。

    Creates the semantically-dependent vectors (of all roles) for all mentions
    in a specific entity cluster.

    :param entity_cluster: a Cluster object which contains EntityMention objects.
    :param event_clusters: current predicted event clusters (a list)
    :param model: CDCorefScorer object
    :param device: Pytorch device
    :return: No return.
        But for each entity mention in *entity_cluster*, the_entity_mention.arg0_vec/arg1_vec/time_vec/loc_vec are setted.

    """
    for entity_mention in entity_cluster.mentions.values():
        entity_mention.arg0_vec = torch.zeros(model.word_embed_dim + model.char_hidden_dim, requires_grad=False).to(device).view(1, -1)
        entity_mention.arg1_vec = torch.zeros(model.word_embed_dim + model.char_hidden_dim, requires_grad=False).to(device).view(1, -1)
        entity_mention.time_vec = torch.zeros(model.word_embed_dim + model.char_hidden_dim, requires_grad=False).to(device).view(1, -1)
        entity_mention.loc_vec = torch.zeros(model.word_embed_dim + model.char_hidden_dim, requires_grad=False).to(device).view(1, -1)
        predicates_dict = entity_mention.predicates
        for predicate_id, rel in predicates_dict.items():
            if rel == 'A0':
                entity_mention.arg0_vec = find_mention_cluster_vec(predicate_id[1], event_clusters).to(device)
            elif rel == 'A1':
                entity_mention.arg1_vec = find_mention_cluster_vec(predicate_id[1], event_clusters).to(device)
            elif rel == 'AM-TMP':
                entity_mention.time_vec = find_mention_cluster_vec(predicate_id[1], event_clusters).to(device)
            elif rel == 'AM-LOC':
                entity_mention.loc_vec = find_mention_cluster_vec(predicate_id[1], event_clusters).to(device)


def update_lexical_vectors(clusters: List[Cluster], model: CDCorefScorer,
                           device: torch.cuda.device, is_event: bool, requires_grad: bool):
    """
    For each cluster in *clusters*, this function set it's lexical vector (the_cluster.lex_vec).

    the_cluster.lex_vec = average(簇内每个指称的指称向量s(m)).

    :param clusters: list of Cluster objects (event/entity clsuters)
    :param model: It should be an event model if clusters are event clusters (and
        the same with entities)
    :param device: Pytorch device
    :param is_event: True, if *clusters* are event clusters; False, if *clusters* are entity clusters.
    :param requires_grad: True if tensors require gradients (for training time) , and
     False for inference time.
    :return: no return, but set cluster.lex_vec
    """
    for cluster in clusters:
        if is_event:
            lex_vec = create_event_cluster_bow_lexical_vec(cluster, model, device, use_char_embeds=True, requires_grad=requires_grad)
        else:
            lex_vec = create_entity_cluster_bow_lexical_vec(cluster, model, device, use_char_embeds=True, requires_grad=requires_grad)
        cluster.lex_vec = lex_vec


def update_args_feature_vectors(clusters: List[Cluster], other_clusters: List[Cluster],
                                model: CDCorefScorer, device: torch.cuda.device, is_event: bool) -> None:
    """

    根据*other_clusters* (实体/事件簇)，更新*clusters* (事件/实体簇)中每个事件/实体指称的语义依存向量 。

    Based on *other_cluster* (entity/event clusters), update the dependency
    vector (includes: arg0, arg1, time, loc) of each mention in *clusters* (event/entity clusters).

    That is V->d(m_e) or E->d(m_v).

    :param clusters: A list of event/entity clusters.
    :param other_clusters: A list of entity/event clusters.
    :param model: event/entity model (should be according to clusters parameter)
    :param device: Pytorch device
    :param is_event: True, if *clusters* are event clusters; False, if *clusters* are entity clusters.
    :return: No return.
        But for each  mention in *cluster*, the_mention.arg0_vec/arg1_vec/time_vec/loc_vec are setted.
    """
    for cluster in clusters:
        if is_event:
            # 基于实体簇，更新事件指称的dependency vector
            create_event_cluster_bow_arg_vec(cluster, other_clusters, model, device)
        else:
            # 基于事件簇，更新实体指称的dependency vector
            create_entity_cluster_bow_predicate_vec(cluster, other_clusters, model, device)


def generate_cluster_pairs(clusters: List[Cluster], is_train) -> Tuple[
    List[Union[Tuple[Cluster, Cluster, float], Tuple[Cluster, Cluster]]],
    List[Tuple[Cluster, Cluster]]
]:
    """
    Given list of clusters, this function generates candidate cluster pairs:
        - If *is_train* is true, this function returns tuple (train_pairs, test_pairs).
        - If *is_train* is false, this function returns tuple (test_pairs, [])

    and
        - train_pairs is a list of tuple，one tuple represents one cluster pair. 
          The tuple likes (cluster1, cluster2, true score).
          train_pairs includes of all possible cluster pairs, if len(clusters) <= 300.
          train_pairs includes of under-sampled cluster pairs, if len(clusters) > 300.
        - test_pairs is a list of tuple, one tuple represents one cluster pair. 
          The tuple likes (cluster1, cluster2).
          test_pairs includes all possible cluster pairs.

    :param clusters: current clusters
    :param is_train: True if the function generates candidate cluster pairs for training time
        , and False for inference time (without under-sampling)
    :return: (train_pairs, test_pairs) or (test_pairs, [])
    """
    logging.info('Generating cluster pairs...')
    logging.info('Initial number of clusters = {}'.format(len(clusters)))

    if is_train:
        positive_pairs_count = 0
        negative_pairs_count = 0
        train_pairs = []  # 用于训练的候选簇对，带共指得分
        test_pairs = []   # 用于测试的候选簇对，不带共指得分
        # 判断train_pairs是否需要下采样
        use_under_sampling = True if len(clusters) > 300 else False
        if len(clusters) < 500:
            p = 0.7
        else:
            p = 0.6
        if use_under_sampling:
            logging.info('Using under sampling with p = {}'.format(p))
        # 遍历所有簇对
        for cluster_1 in clusters:
            for cluster_2 in clusters:
                if cluster_1 != cluster_2:
                    q = calc_q(cluster_1, cluster_2)
                    if (cluster_1, cluster_2, q) not in train_pairs and (cluster_2, cluster_1, q) not in train_pairs:
                        # 把当前簇对添加到train_pairs
                        add_to_training = not use_under_sampling
                        if q > 0:
                            add_to_training = True
                            positive_pairs_count += 1
                        if q == 0 and random.random() < p:
                            add_to_training = True
                            negative_pairs_count += 1
                        if add_to_training:
                            train_pairs.append((cluster_1, cluster_2, q))
                        # 把当前簇对添加到test_pairs
                        test_pairs.append((cluster_1, cluster_2))
        return train_pairs, test_pairs
    else:
        test_pairs = []  # 用于测试的候选簇对，不带共指得分
        # 遍历所有簇对
        for cluster_1 in clusters:
            for cluster_2 in clusters:
                if cluster_1 != cluster_2:
                    if (cluster_1, cluster_2) not in test_pairs and (cluster_2, cluster_1) not in test_pairs:
                        # 把当前簇对添加到test_pairs
                        test_pairs.append((cluster_1, cluster_2))
        return test_pairs, []


def get_mention_span_rep(mention: Mention, device: torch.cuda.device, model: CDCorefScorer,
                         docs: Dict[str, Document], is_event: bool, requires_grad: bool) -> torch.Tensor:
    """
    For *mention*, this function:
        - calc it's span text vector s(m) = average(word embedding of each word in the mention)
        - calc it's context vector c(m) = elmo embedding of mention head.
        - set the_mention.span_rep = (c(m) ; s(m))

    :param mention: an Mention object (either an EventMention or an EntityMention)
    :param device: Pytorch device
    :param model: CDCorefScorer object, should be in the same type as the mention
    :param docs: the current topic's documents
    :param is_event: True if mention is an event mention and False if it is an entity mention
    :param requires_grad: True if tensors require gradients (for training time) , and
        False for inference time.
    :return: The span representation of *mention*. It is a tensor with size (1, 1374).
    """
    # 1. get the context vector c(m)
    context_vec = mention.head_elmo_embeddings.to(device).view(1, -1)

    # 2. get the span text vector s(m)
    span_vec: torch.Tensor = torch.zeros(model.word_embed_dim+model.char_hidden_dim, requires_grad=requires_grad).to(device).view(1, -1)
    if is_event:
        head = mention.mention_head
        words_vec = find_word_embed(head, model, device)
        chars_vec = get_char_embed(head, model, device)
        span_vec = torch.cat([words_vec, chars_vec], 1)
    else:
        words_vec = torch.zeros(model.word_embed_dim, requires_grad=requires_grad).to(device).view(1, -1)
        word_vec_list = [find_word_embed(token, model, device) for token in mention.get_tokens()
                         if not is_stop(token)]
        for word_vec in word_vec_list:
            words_vec = words_vec + word_vec
            """
            改bug，mention_bow += mention_word_tensor，改成mention_bow = mention_bow + mention_word_tensor
            (后来命名改了，但就是这个意思)
            问题所在：mention_bow初始没问题，但循环中的+=是inplace操作，所以_version非0，产生了问题
            """
        chars_vec = get_char_embed(mention.mention_str, model, device)
        if len(word_vec_list) > 0:
            words_vec = words_vec / float(len(word_vec_list))
        span_vec = torch.cat([words_vec, chars_vec], 1)

    # 3. mention span rep = (c(m);s(m))
    mention_span_rep = torch.cat([context_vec, span_vec], 1)
    if requires_grad:
        if not mention_span_rep.requires_grad:
            logging.info('mention_span_rep does not require grad ! (warning)')
    else:
        mention_span_rep = mention_span_rep.detach()
    return mention_span_rep


def create_mention_span_representations(
        mentions: List[Mention], model: CDCorefScorer,
        device: torch.cuda.device, topic_docs: dict,
        is_event: bool, requires_grad: bool
) -> None:
    """
    For each mention in *mentions*:
        - calc their span text vector s(m) = average(word embedding of each word in the mention)
        - calc their context vector c(m) = elmo embedding of mention head.
        - set each_mention.span_rep = (c(m) ; s(m))
    Creates for a set of mentions their context and span text vectors.

    :param mentions: a list of EventMention objects (or a list of EntityMention objects)
    :param model: CDCorefScorer object, should be in the same type as the mentions
    :param device: Pytorch device
    :param topic_docs: the current topic's documents, like {'45_6ecb': Document object,...}
    :param is_event: True if mention is an event mention and False if it is an entity mention
    :param requires_grad: True if tensors require gradients (for training time) , and False for inference time.
    :return: No return. But for each mention in *mentions*, each_mention.span_rep are updated.
    """
    for mention in mentions:
        mention.span_rep = get_mention_span_rep(mention, device, model, topic_docs, is_event, requires_grad)


def mention_pair_to_model_input(pair, model, device, topic_docs, is_event, requires_grad,
                                 use_args_feats, use_binary_feats, other_clusters):
    """
    Given a pair of mentions (m_i, m_j), this function returns the mention-pair
    representation v_i,j (which is the input to the network).

    v_i,j = (v(m_i); v(m_j); v(m_i)-v(m_j); v(m_i)*v(m_j); f(i,j))
        - if model.use_mult is False, delete v(m_i)*v(m_j).
        - if model.use_diff is False, delete v(m_i)-v(m_j).
        - if use_binary_feats is False, delete f(i,j).

    v(m) = (s(m); d(m))
        - if use_args_fears is False, delete d(m)

    :param pair: a tuple of two Mention objects (should be of the same type -
        events or entities)
    :param model: CDCorefScorer object (should be in the same type as pair -
        event or entity Model)
    :param device: Pytorch device
    :param topic_docs: the current topic's documents
    :param is_event: True if pair is an event mention pair and False if it's
        an entity mention pair.
    :param requires_grad: True if tensors require gradients (for training time) ,
        and False for inference time.
    :param use_args_feats: whether to use the semantically-dependent mention
        vectors or to ablate them.
    :param use_binary_feats: whether to use the binary coreference features or to
        ablate them.
    :param other_clusters: should be the current event clusters if pair is an
        entity mention pair and vice versa.
    :return: the mention-pair representation - a tensor of size (1,X), when X =
        8522 in the full joint model (without any ablation)
    """
    mention_1 = pair[0]
    mention_2 = pair[1]

    # create span representation
    if requires_grad:
        mention_1.span_rep = get_mention_span_rep(mention_1, device, model, topic_docs,
                                                  is_event, requires_grad)
        mention_2.span_rep = get_mention_span_rep(mention_2, device, model, topic_docs,
                                                  is_event, requires_grad)
    span_rep_1 = mention_1.span_rep
    span_rep_2 = mention_2.span_rep

    if use_args_feats:
        mention_1_tensor = torch.cat([span_rep_1, mention_1.arg0_vec, mention_1.arg1_vec,
                                      mention_1.loc_vec, mention_1.time_vec], 1)
        mention_2_tensor = torch.cat([span_rep_2, mention_2.arg0_vec,mention_2.arg1_vec,
                                      mention_2.loc_vec,mention_2.time_vec], 1)

    else:
        mention_1_tensor = span_rep_1
        mention_2_tensor = span_rep_2

    # mention_1_tensor 有问题
    if model.use_mult and model.use_diff:
        mention_pair_tensor = torch.cat([mention_1_tensor, mention_2_tensor,
                                         mention_1_tensor - mention_2_tensor,
                                         mention_1_tensor * mention_2_tensor], 1)
    elif model.use_mult:
        mention_pair_tensor = torch.cat([mention_1_tensor, mention_2_tensor,
                                         mention_1_tensor * mention_2_tensor], 1)
    elif model.use_diff:
        mention_pair_tensor = torch.cat([mention_1_tensor, mention_2_tensor,
                                         mention_1_tensor - mention_2_tensor], 1)

    if use_binary_feats:
        if is_event:
            binary_feats = create_args_features_vec(mention_1, mention_2, other_clusters,
                                                    device, model)
        else:
            binary_feats = create_predicates_features_vec(mention_1, mention_2, other_clusters,
                                                          device, model)

        mention_pair_tensor = torch.cat([mention_pair_tensor,binary_feats], 1)

    mention_pair_tensor = mention_pair_tensor.to(device)

    return mention_pair_tensor


def train_pairs_batch_to_model_input(batch_pairs, model, device, topic_docs, is_event,
                                      use_args_feats, use_binary_feats, other_clusters):
    '''
    Creates input tensors (mention pair representations) to all mention pairs in the batch
    (for training time).

    :param batch_pairs: a list of mention pairs (in the size of the batch)
    :param model: CDCorefScorer object (should be in the same type as batch_pairs
     - event or entity Model)
    :param device: Pytorch device
    :param topic_docs: the current topic's documents
    :param is_event:  True if pairs are event mention pairs and False if they are
    entity mention pairs.
    :param use_args_feats: whether to use the semantically-dependent mention vectors or to ablate
    them.
    :param use_binary_feats: whether to use the binary coreference features or to ablate
    them.
    :param other_clusters: should be the current event clusters if batch_pairs are entity mention
     pairs and vice versa.
    :return: batch_pairs_tensor - a tensor of the mention pair representations
    according to the batch size, q_pairs_tensor - a tensor of the pairs' gold labels
    '''
    tensors_list = []
    q_list = []

    for pair in batch_pairs:
        # v_i,j = (v(m_i); v(m_j); v(m_i) * v(m_j); f(i, j))
        mention_pair_tensor = mention_pair_to_model_input(pair, model, device, topic_docs,
                                                          is_event, requires_grad=True,
                                                          use_args_feats=use_args_feats,
                                                          use_binary_feats=use_binary_feats,
                                                          other_clusters=other_clusters)

        if not mention_pair_tensor.requires_grad:
            logging.info('mention_pair_tensor does not require grad ! (warning)')

        tensors_list.append(mention_pair_tensor)
        if len(pair) == 2:
            q = 1.0 if pair[0].gold_tag == pair[1].gold_tag else 0.0
            q_tensor = float_to_tensor(q, device)
            q_list.append(q_tensor)
        else:
            q = pair[2]
            q_tensor = float_to_tensor(q, device)
            q_list.append(q_tensor)


    batch_pairs_tensor = torch.cat(tensors_list, 0)
    q_pairs_tensor = torch.cat(q_list, 0)

    return batch_pairs_tensor, q_pairs_tensor


def train(cluster_pairs, model, optimizer, loss_function, device, topic_docs, epoch,
          topics_counter, topics_num, config_dict, is_event, other_clusters):
    '''
    Trains a model using a given set of cluster pairs, a specific optimizer and a loss function.
    The model is trained on all mention pairs between each cluster pair.

    :param cluster_pairs: list of clusters pairs
    :param model: CDCorefModel object
    :param optimizer: Pytorch optimizer
    :param loss_function: Pytorch loss function
    :param device: Pytorch device
    :param topic_docs: the current topic's documents
    :param epoch: current epoch
    :param topics_counter: current topic number
    :param topics_num: total number of topics
    :param config_dict: configuration dictionary, stores the configuration of the experiment
    :param is_event: True, if model is an event model and False if it's an entity model
    :param other_clusters: should be the current event clusters if the function trains
     an entity model and vice versa.
    '''
    batch_size = config_dict["batch_size"]
    mode = 'Event' if is_event else 'Entity'
    retain_graph = False
    epochs = config_dict["regressor_epochs"]
    random.shuffle(cluster_pairs)

    # creates mention pairs and their true labels (creates max 100,000 mention pairs - due to memory constrains)
    pairs = cluster_pairs_to_mention_pairs(cluster_pairs)
    pairs_2 = cluster_pairs_to_mention_pairs_2(cluster_pairs)

    """
    假设有三个cluster,分别为cluster1, cluster2, cluster3. cluster1.mentions  {"1" : mention1; "2" : mention2}
    cluster2.mentions = {"1" : mention3; "2" : mention4} cluster3.mentions = {"1" : mention5; "2" : mention6}
    
    cluster_pairs = [(cluster1, cluster2),(cluster1, cluster3),(cluster2, cluster3)]
    
    pairs = [(mention1,mention3),(mention1,mention4),(mention2,mention3),(mention2,mention4),
             (mention1,mention5),(mention1,mention6),(mention2,mention5),(mention2,mention6),
             (mention3,mention5),(mention3,mention6),(mention4,mention5),(mention4,mention6)]
             
    pairs_2 = [(mention1,mention3, 1.0), (mention1,mention4, 1.0),(mention2,mention3, 0.0),....]
    """
    pairs.extend(pairs_2)
    random.shuffle(pairs)

    for reg_epoch in range(0, epochs):
        samples_count = 0
        batches_count = 0
        total_loss = 0
        batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size) if i + batch_size < len(pairs)]
        for batch_pairs in batches:
            samples_count += batch_size
            batches_count += 1
            batch_tensor, q_tensor = train_pairs_batch_to_model_input(batch_pairs, model,
                                                                      device, topic_docs, is_event,
                                                                      config_dict["use_args_feats"],
                                                                      config_dict["use_binary_feats"],
                                                                      other_clusters)

            model.zero_grad()
            output = model(batch_tensor)
            loss = loss_function(output, q_tensor)
            loss.backward(retain_graph=retain_graph)
            optimizer.step()
            total_loss += loss.item()

            if samples_count % config_dict["log_interval"] == 0:
                logging.info('epoch {}, topic {}/{} - {} model [{}/{} ({:.0f}%)]  Loss: {:.6f}'.format(
                    epoch, topics_counter, topics_num, mode, samples_count, len(pairs),
                    100. * samples_count / len(pairs), (total_loss/float(batches_count)))
                )

            del batch_tensor, q_tensor


def cluster_pair_to_mention_pair(pair):
    '''
    Given a cluster pair, the function extracts all the mention pairs between the two clusters
    :param pair: a cluster pair (tuple of two Cluster objects)
    :return: a list contains tuples of Mention object pairs (EventMention/EntityMention)
    '''
    mention_pairs = []
    cluster_1 = pair[0]
    cluster_2 = pair[1]

    c1_mentions = cluster_1.mentions.values()
    c2_mentions = cluster_2.mentions.values()


    for mention_1 in c1_mentions:
        for mention_2 in c2_mentions:
            mention_pairs.append((mention_1, mention_2))

    return mention_pairs


def cluster_pairs_to_mention_pairs(cluster_pairs):
    '''
    Generates all mention pairs between all cluster pairs

    :param cluster_pairs: cluster pairs (tuples of two Cluster objects)
    :return: a list contains tuples of Mention object pairs (EventMention/EntityMention)
    '''
    th = 100000
    mention_pairs = []

    for pair in cluster_pairs:

        mention_pairs.extend(cluster_pair_to_mention_pair(pair))

        if len(mention_pairs) > th: # up to 100,000 pairs (due to memory constrains)
            break

    return mention_pairs

def cluster_pairs_to_mention_pairs_2(cluster_pairs):
    '''
    按照我们的方法生成新的部分数据集

    :param cluster_pairs: cluster pairs (tuples of two Cluster objects)
    :return: a list contains tuples of Mention object pairs and its score (EventMention/EntityMention)
    '''
    mention_pairs = []
    cluster_list = []
    mentions_list = []
    new_positive_example_count = 0
    new_negative_example_count = 0

    for pair in cluster_pairs:
        cluster_1 = pair[0]
        cluster_2 = pair[1]
        if cluster_1 not in cluster_list:
            cluster_list.append(cluster_1)
        if cluster_2 not in cluster_list:
            cluster_list.append(cluster_2)

    for cluster in cluster_list:
        cluster_mentions = list(cluster.mentions.values())
        mentions_list.extend(cluster_mentions)

    cluster_gold_tag_set = set()
    for mention in mentions_list:
        cluster_gold_tag_set.add(mention.gold_tag)

    file_test_modify = open('output_test/test_modify_2.txt', 'a+') # 检查修改后的结果，目前仅正例
    for cluster_gold_tag in cluster_gold_tag_set:
        # 获得真实的共指指称
        coref_mentions_1 = []
        for mention in mentions_list:
            if mention.gold_tag == cluster_gold_tag:
                coref_mentions_1.append(copy.deepcopy(mention))

        if isinstance(coref_mentions_1[0], EntityMention):
            if coref_mentions_1[0].mention_type == "HUM":
                # 判断mention的mention_str是否为人名
                # 生成正例
                th = 50000
                from src.all_models.name_generator import gen_two_words
                new_mention_str_1 = gen_two_words()
                for coref_mention in coref_mentions_1:
                    file_test_modify.write(coref_mention.mention_str)
                    if word_is_name(coref_mention.mention_str):
                        coref_mention.mention_str = new_mention_str_1
                        file_test_modify.write("[" + coref_mention.mention_str + "]; ")
                    else:
                        file_test_modify.write("; ")

                for cm1 in coref_mentions_1:
                    for cm2 in coref_mentions_1:
                        if cm1 is not cm2:
                            mention_pairs.append((cm1, cm2, 1.0))
                            new_positive_example_count += 1
                        if new_positive_example_count > th:
                            break

                # # 生成反例
                # th = 50000
                # new_mention_str_2 = new_mention_str_1
                # coref_mentions_2 = copy.deepcopy(coref_mentions_1)
                # while new_mention_str_2 == new_mention_str_1:
                #     new_mention_str_2 = gen_two_words()
                #
                # # # 生成反例时 只修改除代词词性以外的mention
                # # while True:
                # #     choosed_mention = random.choice(coref_mentions_2)
                # #     if not word_is_pron(choosed_mention.mention_head_lemma):
                # #         choosed_mention.mention_str = new_mention_str_2
                # #         break
                #
                # random.choice(coref_mentions_2).mention_str = new_mention_str_2# 生成反例时 随机选取一例修改 不管它的词性
                # for cm1 in coref_mentions_2:
                #     if cm1.mention_str == new_mention_str_2:
                #         for cm2 in coref_mentions_2:
                #             if cm1 is not cm2:
                #                 mention_pairs.append((cm1, cm2, 0.0))
                #                 mention_pairs.append((cm2, cm1, 0.0))
                #                 new_negative_example_count += 1
                #             if new_negative_example_count > th:
                #                 break
                file_test_modify.write("\n\n\n")

            elif coref_mentions_1[0].mention_type == "NON":
                pass

            elif coref_mentions_1[0].mention_type == "LOC":
                pass

            elif coref_mentions_1[0].mention_type == "TIM":
                pass

        elif isinstance(coref_mentions_1[0], EventMention):
            pass

    file_test_modify.write("------------------------------------\n")

    file_test_modify.close()
    return mention_pairs

def word_is_name(word):
    """
    1.mention_str含有数字或"."，则不是人名
    2.mention_str的首个字符若是字母，则可能是人名转3；若不是字母，则不是人名.
    3.mention_str含有一个单词：若不在英文词典，则是人名；若在英文词典，则可能是->查常用姓名表，若在则是，不在则不是。
    4.mention_str一个以上单词：整体在英文词典，则不是；整体不在英文词典，则可能是->取str的前两个单词，如果这两个单词在词典内都搜不到，则是人名；
      如果这两个单词中的任意一个在词典中搜到了->查询第一个单词是否在first_name表，第二个单词是否在last_name表，有一个在则是人名；反之，不是人名。

    缺陷：Andrew Luck / Andrew 会被认定为人名， Luck不会被认定为人名
    """
    import re
    import json

    if bool(re.search(r'\d', word)): # 如果字符串含有数字 不是人名
        return False
    if '.' in word:
        return False

    with open("data/external/human_name/english_dictionary.json", 'r') as f:
        en_dic = json.load(f)
    with open("data/external/human_name/first_names.all.txt", "r") as file_frist_name:
        first_name_list = file_frist_name.read().splitlines()
    with open("data/external/human_name/last_names.all.txt", "r") as file_last_name:
        last_name_list = file_last_name.read().splitlines()

    if (word[0] >= "A" and word[0] <= "Z") or (word[0] >= "a" and word[0] <= "z"):
        word_list = word.split(' ')
        if len(word_list) == 1:  # 如果只有字符串只有一个单词，这个单词在词典里搜不到，那么就认定是一个人名
            if word_list[0] not in en_dic[word_list[0][0]] and word_list[0].lower() not in en_dic[word_list[0][0].lower()]:
                return True
            else:  # 如果能找到，那么可能是一个人名。在常用姓/名库内能找到，则是；找不到，则不是。  为什么用小写？因为姓名库里存储的都是小写。
                if word_list[0].lower() in first_name_list:
                    return True
                elif word_list[0].lower() in last_name_list:
                    return True
                else:
                    return False
        else:  # 如果字符串有2个单词
            # 1.这两个词整体能找到，则一定不是人名；整体找不到，则可能是人名,转2。
            if word in en_dic[word[0]]:
                return False
            else:
                # 2.把整体拆开为两个词。两个词都搜不到，则是人名；两个词中任意一个能搜到，则可能是人名，转3
                if word_list[0] not in en_dic[word_list[0][0]] \
                        and word_list[0].lower() not in en_dic[word_list[0][0].lower()] \
                        and word_list[1] not in en_dic[word_list[1][0]] \
                        and word_list[1].lower() not in en_dic[word_list[1][0].lower()]:
                    return True
                else:
                    # 3.如果在词典中搜到的这个词，在常用姓名库内 则是人名，反之不是。
                    if word_list[0].lower() in first_name_list or word_list[1].lower() in last_name_list:
                        return True
                    else:
                        return False
    else:
        return False # 如果mention_str的开头不是英文字母，它就不是一个英文名


def word_is_pron(word):
    """
    判断单词是否为代词。 判别范围 人称代词+反身代词+物主代词
    """
    pron_list = ["I", "me", "my", "mine", "myself",
                 "we", "us", "our", "ours", "ourselves",
                 "you", "your", "yours", "yourself", "yourselves",
                 "he", "him", "his", "himself",
                 "she", "her", "hers", "herself",
                 "it", "its", "itself",
                 "they", "them", "their", "theirs", "themselves"
                 ]
    if word not in pron_list:
        return False
    else:
        return True

def test_pairs_batch_to_model_input(batch_pairs, model, device, topic_docs, is_event,
                                     use_args_feats, use_binary_feats, other_clusters):

    '''
    Creates input tensors (mention pair representations) for all mention pairs in the batch
    (for inference time).
    :param batch_pairs: a list of mention pairs (in the size of the batch)
    :param model: CDCorefScorer object (should be in the same type as batch_pairs
     - event or entity Model)
    :param device: Pytorch device
    :param topic_docs: the current topic's documents
    :param is_event:  True if pairs are event mention pairs and False if they are
    entity mention pairs.
    :param use_args_feats: whether to use the semantically-dependent mention vectors or to ablate
    them.
    :param use_binary_feats: whether to use the binary coreference features or to ablate
    them.
    :param other_clusters: should be the current event clusters if batch_pairs are entity mention
     pairs and vice versa.
    :return: batch_pairs_tensor - a tensor of the mention pair representations
    according to the batch size, q_pairs_tensor - a tensor of the pairs' gold labels

    '''
    tensors_list = []
    for pair in batch_pairs:
        mention_pair_tensor = mention_pair_to_model_input(pair, model, device, topic_docs,
                                                          is_event, requires_grad=False,
                                                          use_args_feats=use_args_feats,
                                                          use_binary_feats=use_binary_feats,
                                                          other_clusters=other_clusters)
        tensors_list.append(mention_pair_tensor)

    batch_pairs_tensor = torch.cat(tensors_list, 0)

    return batch_pairs_tensor


def get_batches(mention_pairs, batch_size):
    """
    Splits the mention pairs to batches (specifically this function used during inference time)

    :param mention_pairs: a list contains a tuples of mention pairs
    :param batch_size: the batch size (integer)
    :return: list of lists, when each inner list contains each batch's pairs
    """
    batches = [mention_pairs[i:i + batch_size] for i in
               range(0, len(mention_pairs),batch_size) if i + batch_size < len(mention_pairs)]
    diff = len(mention_pairs) - len(batches)*batch_size
    if diff > 0:
        batches.append(mention_pairs[-diff:])

    return batches

def key_with_max_val(d):
    """ a) creates a list of the dict's keys and values;
        b) returns the key with the max value and the max value"""
    v = list(d.values()) #scores
    k = list(d.keys()) #pairs

    np_scores = np.asarray(v)
    best_ix = np.argmax(np_scores)
    best_score = np_scores[best_ix]

    return k[v.index(best_score)], best_score


def merge_clusters(pair_to_merge: Tuple[Cluster, Cluster],
                   clusters: List[Cluster], other_clusters: List[Cluster],
                   is_event, model, device, topic_docs: Dict[str, Document],
                   candidate_pairs: Dict[Tuple[Cluster, Cluster], float],
                   use_args_feats, use_binary_feats) -> None:
    """
    This function:
        - 基于 *pair_to_merge* 中的两个旧簇, 进行合并, 创建新簇, 并计算新簇的mentions, lex_vec,
          arg0_vec, arg1_vec, time_vec, loc_vec.
        - 更新当前簇*clusters*：
          1. 删除旧簇
          2. 添加新簇
        - 更新候选簇对*candidate_pairs*:
          1. 删除旧簇对(所有涉及旧簇的簇对);
          2. 添加新簇对(新簇和当前每个簇各组成一个新簇对).

    :param pair_to_merge: a tuple of two Cluster objects that were chosen to get merged.
    :param clusters: 所有此类簇。current event/entity clusters (of the same type of pair_to_merge)
    :param other_clusters: 所有它类簇。should be the current event clusters if clusters are entity clusters and vice versa.
    :param is_event: True if pair_to_merge is an event pair  and False if they it's an entity pair.
    :param model: CDCorefModel object
    :param device: Pytorch device object
    :param topic_docs: current topic's documents
    :param candidate_pairs: 所有候选簇对及其得分。dictionary contains the current candidate cluster pairs
    :param use_args_feats: whether to use the semantically-dependent mention vectors or to ablate them.
    :param use_binary_feats: whether to use the binary coreference features or to ablate them.
    :return: No return. *clusters* updated, *candidate_pairs* updated.
    """
    cluster_i = pair_to_merge[0]
    cluster_j = pair_to_merge[1]

    # 新簇的创建
    new_cluster = Cluster(is_event)

    # 新簇的指称
    new_cluster.mentions.update(cluster_j.mentions)
    new_cluster.mentions.update(cluster_i.mentions)

    # 候选簇对:删除旧簇对
    keys_pairs_dict: List[Tuple[Cluster, Cluster]] = list(candidate_pairs.keys())
    for pair in keys_pairs_dict:
        cluster_pair = (pair[0], pair[1])
        if cluster_i in cluster_pair or cluster_j in cluster_pair:
            del candidate_pairs[pair]

    # 本类簇列表:删除旧簇
    clusters.remove(cluster_i)
    clusters.remove(cluster_j)
    # 本类簇列表:添加新簇
    clusters.append(new_cluster)

    # 新簇的向量
    if is_event:
        lex_vec = create_event_cluster_bow_lexical_vec(new_cluster, model, device,
                                                       use_char_embeds=True,
                                                       requires_grad=False)
    else:
        lex_vec = create_entity_cluster_bow_lexical_vec(new_cluster, model, device,
                                                        use_char_embeds=True,
                                                        requires_grad=False)
    new_cluster.lex_vec = lex_vec

    # 新簇的语义依存向量 create arguments features for the new cluster
    update_args_feature_vectors([new_cluster], other_clusters, model, device, is_event)

    # 候选簇对：添加新簇对
    new_pairs = []
    for cluster in clusters:
        if cluster != new_cluster:
            new_pairs.append((cluster, new_cluster))
    # create scores for the new pairs
    for pair in new_pairs:
        pair_score = assign_score(pair, model, device, topic_docs, is_event,
                                  use_args_feats, use_binary_feats, other_clusters)
        candidate_pairs[pair] = pair_score


def assign_score(cluster_pair, model, device, topic_docs, is_event, use_args_feats,
                 use_binary_feats, other_clusters):
    """
    Assigns coreference (or quality of merge) score to a cluster pair by averaging the mention-pair
    scores predicted by the model.

    :param cluster_pair: a tuple of two Cluster objects
    :param model: CDCorefScorer object
    :param device: Pytorch device
    :param topic_docs: current topic's documents
    :param is_event: True if cluster_pair is an event pair and False if it's an entity pair
    :param use_args_feats: whether to use the semantically-dependent mention vectors or to ablate
        them.
    :param use_binary_feats: whether to use the binary coreference features or to ablate
        them.
    :param other_clusters: should be the current event clusters if cluster_pair is an entity pair
        and vice versa.
    :return: The average mention pairwise score
    """
    mention_pairs = cluster_pair_to_mention_pair(cluster_pair)
    batches = get_batches(mention_pairs, 256)
    pairs_count = 0
    scores_sum = 0
    for batch_pairs in batches:
        batch_tensor = test_pairs_batch_to_model_input(batch_pairs, model, device,
                                                       topic_docs, is_event,
                                                       use_args_feats=use_args_feats,
                                                       use_binary_feats=use_binary_feats,
                                                       other_clusters=other_clusters)

        model_scores = model(batch_tensor).detach().cpu().numpy()
        scores_sum += float(np.sum(model_scores))
        pairs_count += len(model_scores)

        del batch_tensor

    return scores_sum/float(pairs_count)


def merge(clusters: List[Cluster],
          pairs: List[Tuple[Cluster, Cluster]], other_clusters: List[Cluster],
          model: CDCorefScorer, device: torch.cuda.device,
          topic_docs, epoch, topics_counter,
          topics_num, threshold, is_event, use_args_feats, use_binary_feats) -> None:
    """
    Merges cluster pairs in agglomerative manner till it reaches a pre-defined
    threshold. In each step, the function merges the cluster pair with the
    highest score, and updates the candidate cluster pairs according to the
    current merge.

    Note that all Cluster objects in *clusters* should have the same type (event
    or entity but not both of them).

    Note that *clusters* are updated and *other_clusters* are fixed during merges.

    :param clusters: a list of event/entity Cluster objects.
    :param pairs: a list of all candidate cluster pairs.
    :param other_clusters: a list of entity/event Cluster objects (with the opposite type to *clusters*) .
    :param model: CDCorefScorer object with the same type as clusters.
    :param device: Pytorch device
    :param topic_docs: current topic's documents
    :param epoch: current epoch (relevant to training)
    :param topics_counter: current topic number
    :param topics_num: total number of topics
    :param threshold: merging threshold
    :param is_event: True if clusters are event clusters and false if they are entity clusters
    :param use_args_feats: whether to use the semantically-dependent mention vectors or to ablate them.
    :param use_binary_feats: whether to use the binary coreference features or to ablate
    :return: No return. But *clusters* are updated.
    """
    logging.info('Initialize cluster pairs scores... ')
    # initializes the pairs-scores dict
    pairs_dict: Dict[Tuple[Cluster, Cluster], float] = {}
    mode = 'event' if is_event else 'entity'
    # 为每个簇对预测得分 init the scores (that the model assigns to the pairs)
    for pair in pairs:
        pair_score = assign_score(pair, model, device, topic_docs, is_event,
                                  use_args_feats, use_binary_feats, other_clusters)
        pairs_dict[pair] = pair_score
    # 迭代的凝聚
    while True:
        # finds max pair (break if we can't find one  - max score < threshold)
        if len(pairs_dict) < 2:
            logging.info('Less the 2 clusters had left, stop merging!')
            break
        max_pair, max_score = key_with_max_val(pairs_dict)
        # 凝聚一下
        if max_score > threshold:
            logging.info('epoch {} topic {}/{} - merge {} clusters with score {} clusters : {} {}'.format(
                epoch, topics_counter, topics_num, mode, str(max_score), str(max_pair[0]),
                str(max_pair[1])))
            merge_clusters(max_pair, clusters, other_clusters, is_event,
                           model, device, topic_docs, pairs_dict,
                           use_args_feats, use_binary_feats)
        # 停止凝聚
        else:
            logging.info('Max score = {} is lower than threshold = {}, stopped merging!'.format(max_score, threshold))
            break


def test_model(clusters, other_clusters, model, device, topic_docs, is_event, epoch,
               topics_counter, topics_num, threshold, use_args_feats,
               use_binary_feats):
    '''
    Runs the inference procedure for a specific model (event/entity model).
    :param clusters: a list of Cluster objects of the same type (event/entity)
    :param other_clusters: a list of Cluster objects with the opposite type to clusters.
    Stays fixed during merging operations on clusters.
    :param model: CDCorefScorer object with the same type as clusters.
    :param device: Pytorch device
    :param topic_docs: current topic's documents
    :param epoch: current epoch (relevant to training)
    :param topics_counter: current topic number
    :param topics_num: total number of topics
    :param threshold: merging threshold
    :param is_event: True if clusters are event clusters and false if they are entity clusters
    :param use_args_feats: whether to use the semantically-dependent mention vectors or to ablate
    them.
    :param use_binary_feats: whether to use the binary coreference features or to ablate them.
    '''

    # updating the semantically - dependent vectors according to other_clusters
    update_args_feature_vectors(clusters, other_clusters, model, device, is_event)

    # generating candidate cluster pairs
    cluster_pairs, _ = generate_cluster_pairs(clusters, is_train=False)

    # merging clusters pairs till reaching a pre-defined threshold
    merge(clusters, cluster_pairs, other_clusters,model, device, topic_docs, epoch,
          topics_counter, topics_num, threshold, is_event, use_args_feats,
          use_binary_feats)

from src.all_models.models import CDCorefScorer
def test_models(
    test_set: Corpus,
    cd_event_model: CDCorefScorer,
    cd_entity_model: CDCorefScorer,
    device: torch.device,
    config_dict: dict, write_clusters: bool, out_dir: str,
    doc_to_entity_mentions: dict,
    analyze_scores: bool
):
    '''
    Runs the inference procedure for both event and entity models, calculates the B-cubed
    score of their predictions.

    :param test_set: 测试集
    :param cd_event_model: CD event coreference model
    :param cd_entity_model: CD entity coreference model
    :param device: Pytorch device
    :param config_dict: 试验配置文件
    :param write_clusters: whether to write predicted clusters to file (for analysis purpose)
    :param out_dir: output files directory
    :param doc_to_entity_mentions: 外部实体共指消解器预测的文档内实体共指结果，用做实体共指簇的初始值
    :param analyze_scores: whether to save representations and Corpus objects for analysis
    :return: B-cubed scores for the predicted event and entity clusters
    '''
    global clusters_count
    clusters_count = 1
    event_errors = []
    entity_errors = []
    all_event_clusters = []
    all_entity_clusters = []

    # 选择文档聚类（就是topic-doc对应关系）
    if config_dict["load_predicted_topics"]:  # 使用外部算法预测的文档聚类
        # test_set是按照ecb真实文档聚类组织的，要按照外部算法预测的文档聚类重新排序组织
        topics = load_predicted_topics(test_set, config_dict)  # use the predicted sub-topics
    else:  # 使用ecb自带的真实文档聚类
        # test_set本来就是按照ecb真实文档聚类组织的，无需处理
        topics = test_set.topics  # use the gold sub-topics

    # while len(topics.keys()) != 2:
    #     if len(topics.keys()) > 2:
    #         topics.popitem()
    #     else:
    #         pass
    #
    # """
    # 减少topic以方便检查代码(仅训练使用)
    # """
    topics_num = len(topics.keys())
    topics_keys = topics.keys()  # topic_keys=[1,2,3,4,...,20]

    epoch = 0  # 训练模型时，每次测试模型过程的epoch没有传递过来，而是在这里新建了一个epoch=0，因此会发生显示不对应的情况，但对想要达到的结果没有影响。
    all_event_mentions = []
    all_entity_mentions = []

    topics_counter = 0
    with torch.no_grad():
        for topic_id in topics_keys:
            topic = topics[topic_id]
            topics_counter += 1

            logging.info('=========================================================================')
            logging.info('Topic {}:'.format(topic_id))

            # 初始化：实体和事件抽取(使用真实事件和实体mention)
            event_mentions, entity_mentions = topic_to_mention_list(
                                                                    topic,
                                                                    is_gold=config_dict["test_use_gold_mentions"]
                                                                    )
            all_event_mentions.extend(event_mentions)  # 把抽取得到的本topic下的事件指称累计到全部事件指称列表
            all_entity_mentions.extend(entity_mentions)  # 把抽取得到的本topic下的实体指称累计到全部实体指称列表

            # 事件和实体的表征
            # create span rep for both entity and event mentions
            create_mention_span_representations(event_mentions, cd_event_model, device,
                                                topic.docs, is_event=True,
                                                requires_grad=False)
            create_mention_span_representations(entity_mentions, cd_entity_model, device,
                                                topic.docs, is_event=False,
                                                requires_grad=False)
            logging.info('number of event mentions : {}'.format(len(event_mentions)))
            logging.info('number of entity mentions : {}'.format(len(entity_mentions)))
            topic.event_mentions = event_mentions
            topic.entity_mentions = entity_mentions

            # initialize within-document entity clusters with the output of within-document system
            wd_entity_clusters = init_entity_wd_clusters(entity_mentions, doc_to_entity_mentions)

            topic_entity_clusters = []
            for doc_id, clusters in wd_entity_clusters.items():
                topic_entity_clusters.extend(clusters)

            # initialize event clusters as singletons
            topic_event_clusters = init_cd(event_mentions, is_event=True)

            # init cluster representation
            update_lexical_vectors(topic_entity_clusters, cd_entity_model, device,
                                   is_event=False, requires_grad=False)
            update_lexical_vectors(topic_event_clusters, cd_event_model, device,
                                   is_event=True, requires_grad=False)

            entity_th = config_dict["entity_merge_threshold"]
            event_th = config_dict["event_merge_threshold"]

            # 初始化结束，开始主循环
            for i in range(1,config_dict["merge_iters"]+1):
                logging.info('Iteration number {}'.format(i))

                # Merge entities
                logging.info('Merge entity clusters...')
                test_model(clusters=topic_entity_clusters, other_clusters=topic_event_clusters,
                           model=cd_entity_model, device=device, topic_docs=topic.docs,is_event=False,epoch=epoch,
                           topics_counter=topics_counter, topics_num=topics_num,
                           threshold=entity_th,
                           use_args_feats=config_dict["use_args_feats"],
                           use_binary_feats=config_dict["use_binary_feats"])
                # Merge events
                logging.info('Merge event clusters...')
                test_model(clusters=topic_event_clusters, other_clusters=topic_entity_clusters,
                           model=cd_event_model,device=device, topic_docs=topic.docs, is_event=True,epoch=epoch,
                           topics_counter=topics_counter, topics_num=topics_num,
                           threshold=event_th,
                           use_args_feats=config_dict["use_args_feats"],
                           use_binary_feats=config_dict["use_binary_feats"])

            set_coref_chain_to_mentions(topic_event_clusters, is_event=True,
                                        is_gold=config_dict["test_use_gold_mentions"],intersect_with_gold=True)
            set_coref_chain_to_mentions(topic_entity_clusters, is_event=False,
                                        is_gold=config_dict["test_use_gold_mentions"],intersect_with_gold=True)

            if write_clusters:
                # Save for analysis
                all_event_clusters.extend(topic_event_clusters)
                all_entity_clusters.extend(topic_entity_clusters)

                with open(os.path.join(out_dir, 'entity_clusters.txt'), 'a') as entity_file_obj:
                    src.shared.eval_utils.write_clusters_to_file(topic_entity_clusters, entity_file_obj, topic_id)
                    entity_errors.extend(collect_errors(topic_entity_clusters, topic_event_clusters, topic.docs,
                                                        is_event=False))

                with open(os.path.join(out_dir, 'event_clusters.txt'), 'a') as event_file_obj:
                    src.shared.eval_utils.write_clusters_to_file(topic_event_clusters, event_file_obj, topic_id)
                    event_errors.extend(collect_errors(topic_event_clusters, topic_entity_clusters, topic.docs,
                                                       is_event=True))

        if write_clusters:
            write_event_coref_results(test_set, out_dir, config_dict)
            write_entity_coref_results(test_set, out_dir, config_dict)
            sample_errors(event_errors, os.path.join(out_dir,'event_errors'))
            sample_errors(entity_errors, os.path.join(out_dir,'entity_errors'))

    if analyze_scores:
        # Save mention representations
        save_mention_representations(all_event_clusters, out_dir, is_event=True)
        save_mention_representations(all_entity_clusters, out_dir, is_event=False)

        # Save topics for analysis
        with open(os.path.join(out_dir,'test_topics'), 'wb') as f:
            cPickle.dump(topics, f)

    if config_dict["test_use_gold_mentions"]:
        event_predicted_lst = [event.cd_coref_chain for event in all_event_mentions]
        true_labels = [event.gold_tag for event in all_event_mentions]
        true_clusters_set = set(true_labels)

        labels_mapping = {}
        for label in true_clusters_set:
            labels_mapping[label] = len(labels_mapping)

        event_gold_lst = [labels_mapping[label] for label in true_labels]
        event_r, event_p, event_b3_f1 = bcubed(event_gold_lst, event_predicted_lst)

        entity_predicted_lst = [entity.cd_coref_chain for entity in all_entity_mentions]
        true_labels = [entity.gold_tag for entity in all_entity_mentions]
        true_clusters_set = set(true_labels)

        labels_mapping = {}
        for label in true_clusters_set:
            labels_mapping[label] = len(labels_mapping)

        entity_gold_lst = [labels_mapping[label] for label in true_labels]
        entity_r, entity_p, entity_b3_f1 = bcubed(entity_gold_lst, entity_predicted_lst)

        return event_b3_f1, entity_b3_f1

    else:
        logging.info('Using predicted mentions, can not calculate CoNLL F1')
        return 0,0


def init_clusters_with_lemma_baseline(mentions, is_event):
    '''
    Initializes clusters for agglomerative clustering with the output of the head lemma baseline
    (used for experiments)
    :param mentions: list of Mention objects (EventMention/EntityMention objects)
    :param is_event: True if mentions are event mentions and False if they are entity mentions
    :return: list of Cluster objects
    '''
    mentions_by_head_lemma = {}
    clusters = []

    for mention in mentions:
        if mention.mention_head_lemma not in mentions_by_head_lemma:
            mentions_by_head_lemma[mention.mention_head_lemma] = []
        mentions_by_head_lemma[mention.mention_head_lemma].append(mention)

    for head_lemma, mentions in mentions_by_head_lemma.items():
        cluster = Cluster(is_event=is_event)
        for mention in mentions:
            cluster.mentions[mention.mention_id] = mention
        clusters.append(cluster)

    return clusters


def mention_data_to_string(mention, other_clusters, is_event,topic_docs):
    '''
    Creates a string representing a mention's data
    :param mention: a Mention object (EventMention/EntityMention)
    :param other_clusters: current entity clusters if mention is an event mention and vice versa
    :param is_event: True if mention is an event mention and False if it's an entity mention.
    :param topic_docs: current topic's documents
    :return: a string representing a mention's data
    '''
    strings = ['mention: {}_{}'.format(mention.mention_str,mention.gold_tag)]
    if is_event:
        if mention.arg0 is not None:
            arg0_cluster = find_mention_cluster(mention.arg0[1], other_clusters)
            gold_arg0_chain = arg0_cluster.mentions[mention.arg0[1]].gold_tag
            strings.append('arg0: {}_{}_{}'.format(mention.arg0[0], arg0_cluster.cluster_id,
                                                   gold_arg0_chain))
        if mention.arg1 is not None:
            arg1_cluster = find_mention_cluster(mention.arg1[1], other_clusters)
            gold_arg1_chain = arg1_cluster.mentions[mention.arg1[1]].gold_tag
            strings.append('arg1: {}_{}_{}'.format(mention.arg1[0], arg1_cluster.cluster_id,
                                                   gold_arg1_chain))
        if mention.amtmp is not None:
            amtmp_cluster = find_mention_cluster(mention.amtmp[1], other_clusters)
            gold_amtmp_chain = amtmp_cluster.mentions[mention.amtmp[1]].gold_tag
            strings.append('amtmp: {}_{}_{}'.format(mention.amtmp[0], amtmp_cluster.cluster_id,
                                                    gold_amtmp_chain))
        if mention.amloc is not None:
            amloc_cluster = find_mention_cluster(mention.amloc[1], other_clusters)
            gold_amloc_chain = amloc_cluster.mentions[mention.amloc[1]].gold_tag
            strings.append('amloc: {}_{}_{}'.format(mention.amloc[0], amloc_cluster.cluster_id,
                                                    gold_amloc_chain))
    else:
        for pred, rel in mention.predicates.items():
            if rel == 'A0':
                arg0_cluster = find_mention_cluster(pred[1], other_clusters)
                gold_arg0_chain = arg0_cluster.mentions[pred[1]].gold_tag
                strings.append('arg0_p: {}_{}_{}'.format(pred[0], arg0_cluster.cluster_id,
                                                         gold_arg0_chain))
            elif rel == 'A1':
                arg1_cluster = find_mention_cluster(pred[1], other_clusters)
                gold_arg1_chain = arg1_cluster.mentions[pred[1]].gold_tag
                strings.append('arg1_p: {}_{}_{}'.format(pred[0], arg1_cluster.cluster_id,
                                                         gold_arg1_chain))
            elif rel == 'AM-TMP':
                amtmp_cluster = find_mention_cluster(pred[1], other_clusters)
                gold_amtmp_chain = amtmp_cluster.mentions[pred[1]].gold_tag
                strings.append('amtmp_p: {}_{}_{}'.format(pred[0], amtmp_cluster.cluster_id,
                                                          gold_amtmp_chain))
            elif rel == 'AM-LOC':
                amloc_cluster = find_mention_cluster(pred[1], other_clusters)
                gold_amloc_chain = amloc_cluster.mentions[pred[1]].gold_tag
                strings.append('amloc_p: {}_{}_{}'.format(pred[0], amloc_cluster.cluster_id,
                                                          gold_amloc_chain))

    strings.append('sent: {}'.format(topic_docs[mention.doc_id].sentences[mention.sent_id].get_raw_sentence()))
    return '\n'.join(strings)


def collect_errors(clusters, other_clusters, topic_docs, is_event):
    '''
    collect event mentions/entity mentions that were clustered incorrectly,
    i.e. where their predicted cluster contained at least 70% of mentions
    that are not in their gold cluster.
    :param clusters: list of Cluster objects of the same type (event/entity clusters)
    :param other_clusters: list of Cluster objects which sets to current entity clusters if clusters are event clusters,
     and vice versa
    :param topic_docs: the current topic's documents
    :param is_event: True if clusters are event clusters and False if they are entity clusters
    :return: set of tuples, when each tuple represents an error.
    '''
    errors = []
    error_ratio = 0.7
    for cluster in clusters:
        mentions_list = []
        for mention in cluster.mentions.values():
            mentions_list.append(mention_data_to_string(mention, other_clusters, is_event, topic_docs))
        cluster_mentions = list(cluster.mentions.values())
        if len(cluster_mentions) > 1:
            for mention_1 in cluster_mentions:
                errors_count = 0
                for mention_2 in cluster_mentions:
                    if mention_1.gold_tag != mention_2.gold_tag:
                        errors_count += 1
                if errors_count/float(len(cluster_mentions)-1) > error_ratio:
                    errors.append((mention_data_to_string(mention_1, other_clusters, is_event, topic_docs)
                                   , mentions_list))

    return errors


def sample_errors(error_list, out_path):
    '''
    Samples 50 errors from error_list
    :param error_list: list of errors collected from each topic
    :param out_path: path to output file
    '''
    random.shuffle(error_list)
    sample = error_list[:50]
    with open(out_path,'w') as f:
        for error in sample:
            f.write('Wrong mention - {}\n'.format(error[0]))
            f.write('cluster: \n')
            for mention in error[1]:
                f.write('{}\n'.format(mention))
                f.write('\n')
            f.write('------------------------------------------------------S')
            f.write('\n')


def mention_to_rep(mention):
    '''
    Returns the mention's representation and its components.
    :param mention: a Mention object (EventMention/EntityMention)
    :return: the mention's representation and its components (tuple of three tensors).
    '''
    span_rep = mention.span_rep
    mention_tensor = torch.squeeze(torch.cat([span_rep, mention.arg0_vec, mention.arg1_vec,
                                  mention.loc_vec, mention.time_vec], 1)).cpu().numpy()
    args_vector = torch.squeeze(torch.cat([mention.arg0_vec, mention.arg1_vec,
                                  mention.loc_vec, mention.time_vec], 1)).cpu().numpy()

    context_vector = mention.head_elmo_embeddings

    return mention_tensor, args_vector , context_vector


def save_mention_representations(clusters, out_dir, is_event):
    '''
    Saves to a pickle file, all mention representations (belong to the test set)
    :param clusters: list of Cluster objects
    :param out_dir: output directory
    :param is_event: True if clusters are event clusters and False if they are entity clusters
    '''
    mention_to_rep_dict = {}
    for cluster in clusters:
        for mention_id, mention in cluster.mentions.items():
            mention_rep = mention_to_rep(mention)
            mention_to_rep_dict[(mention.mention_str, mention.gold_tag)] = mention_rep
    filename = 'event_mentions_to_rep_dict' if is_event else 'entity_mentions_to_rep_dict'
    with open(os.path.join(out_dir, filename), 'wb') as f:
        cPickle.dump(mention_to_rep_dict, f)