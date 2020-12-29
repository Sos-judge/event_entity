import os
import gc
import sys
import time
import math
import json
import spacy
import random
import logging
import argparse
import itertools
import numpy as np
from scorer import *
import _pickle as cPickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from spacy.lang.en import English
# from breakpointAlarm.breakpointAlarm import alarm
from typing import Dict, List, Tuple, Union  # for type hinting
from src.shared.classes import Corpus, Topic, Document, Sentence, Mention, EventMention, EntityMention, Token, Srl_info, Cluster
from src.shared.eval_utils import *
from src.all_models.model_utils import *
from src.all_models.model_factory import factory_load_embeddings, create_model, create_optimizer, create_loss


# parse the arguments in command
parser = argparse.ArgumentParser(description='Training a regressor')
parser.add_argument('--config_path', type=str,
                    help=' The path configuration json file')
parser.add_argument('--out_dir', type=str,
                    help=' The directory to the output folder')
args = parser.parse_args()

# make the output dir
out_dir = args.out_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# 配置logging
import logging
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)
streamHandler = logging.StreamHandler(stream=sys.stdout)  # sys.stderr
streamHandler.setLevel(logging.DEBUG)
fileHandler = logging.FileHandler(filename=os.path.join(out_dir, "train_model.log"), mode='w', encoding='utf8')   # delay=False
fileHandler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt='%(asctime)s\t：%(levelname)s - %(message)s',
    datefmt=None, style='%'
)
streamHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)
rootLogger.addHandler(fileHandler)
rootLogger.addHandler(streamHandler)

# Load json config file
with open(args.config_path, 'r') as js_file:
    config_dict = json.load(js_file)
# copy json config file into output path
with open(os.path.join(args.out_dir,'train_config.json'), "w") as js_file:
    json.dump(config_dict, js_file, indent=4, sort_keys=True)

random.seed(config_dict["random_seed"])
np.random.seed(config_dict["random_seed"])
torch.manual_seed(config_dict["seed"])

if config_dict["gpu_num"] != -1:  # use GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config_dict["gpu_num"])
    args.use_cuda = True
else:  # use CPU
    args.use_cuda = False
args.use_cuda = args.use_cuda and torch.cuda.is_available()
if args.use_cuda:
    torch.cuda.manual_seed(config_dict["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('Training with CUDA')


def train_model(train_set: Corpus, dev_set: Corpus) -> None:
    """
    This function:
        1. Initializes models, optimizers and loss functions,
        2. Then, it runs the training procedure that alternates between entity and
           event training and clustering on the train set.
        3. After each epoch, it runs the inference procedure on the dev set and
           calculates the B-cubed measure and use it to tune the model and its
           hyper-parameters.
        4. Saves the entity and event models that achieved the best B-cubed scores on the dev set.

    :param train_set: The train set.
    :param dev_set: The dev set.
    :return: No return. The entity and event models that achieved the best B-cubed scores on the dev set are saved.
    """

    # loads predicted WD entity coref chains from external tool
    doc_to_entity_mentions = load_entity_wd_clusters(config_dict)

    # loading pre-trained embeddings before creating new models
    factory_load_embeddings(config_dict)

    # create model
    logging.info('Create model')
    device = torch.device("cuda:0" if args.use_cuda else "cpu")

    cd_event_model = create_model(config_dict)
    cd_event_model = cd_event_model.to(device)

    cd_entity_model = create_model(config_dict)
    cd_entity_model = cd_entity_model.to(device)

    # create optimizer
    logging.info('Create optimizer')
    cd_event_optimizer = create_optimizer(config_dict, cd_event_model)
    cd_entity_optimizer = create_optimizer(config_dict, cd_entity_model)

    # create loss function
    logging.info('Create loss function')
    cd_event_loss = create_loss(config_dict)
    cd_entity_loss = create_loss(config_dict)

    topics: Dict[str, Topic] = train_set.topics  # Use the gold sub-topics
    """
    topic dict of train set. ::
    
        {
            '1_ecb': a Topic object,
            '1_ecbplus': a Topic object
        }
    """
    topics_num = len(topics.keys())
    event_best_dev_f1 = 0
    entity_best_dev_f1 = 0
    best_event_epoch = 0
    best_entity_epoch = 0
    patient_counter = 0
    orig_event_th = config_dict["event_merge_threshold"]
    """ original value of config_dict["event_merge_threshold"]    """
    orig_entity_th = config_dict["entity_merge_threshold"]
    """ original value of config_dict["entity_merge_threshold"]    """

    for epoch in range(1, config_dict["epochs"]):  # run the whole data set *epoch* times
        logging.info('Epoch {}:'.format(str(epoch)))

        topics_keys = list(topics.keys())  # ['1_ecb', '3_ecb', '4_ecb', '6_ecb', '7_ecb',...]
        random.shuffle(topics_keys)
        topics_counter = 0
        """ In cur epoch, how many topics has been processed or being processed. """

        # 1. training models on whole train set once (one epoch)
        """ for t ∈ T do   """
        for topic_id in topics_keys:
            topics_counter += 1
            topic = topics[topic_id]

            logging.info('=========================================================================')
            logging.info('Topic {}:'.format(topic_id))
            print('Topic {}:'.format(topic_id))

            # 1.1. extract golden event and entity mention
            event_mentions, entity_mentions = topic_to_mention_list(topic, is_gold=True)

            # 1.2. initialize entity cluster
            if 1:
                entity_clusters: List[Cluster] = []
                """ entity cluster list. """
                # strategy 1: initial entity clusters = singleton clusters.
                if 0:  # we don't use this strategy.
                    entity_clusters = mention_list_to_singleton_cluster_list(entity_mentions, is_event=False)
                # strategy 2: initial entity clusters = gold WD entity coref clusters.
                elif config_dict["train_init_wd_entity_with_gold"]:
                    entity_clusters = mention_list_to_gold_wd_cluster_list(entity_mentions, is_event=False)
                # strategy 3: initial entity clusters = external WD entity coref clusters
                else:
                    entity_clusters = mention_list_to_external_wd_cluster_list(entity_mentions, doc_to_entity_mentions, is_event=False, )
                # calc entity cluster representation
                update_lexical_vectors(entity_clusters, cd_entity_model, device,
                                       is_event=False, requires_grad=False)
            # 1.3. initialize event cluster
            if 1:
                event_clusters = []
                """ event Cluster list.  """
                # strategy 1: initial event clusters = singleton clusters.
                event_clusters = mention_list_to_singleton_cluster_list(event_mentions, is_event=True)
                # calc event cluster representation
                update_lexical_vectors(event_clusters, cd_event_model, device,
                                       is_event=True, requires_grad=False)

            # 1.4.
            entity_th = config_dict["entity_merge_threshold"]
            event_th = config_dict["event_merge_threshold"]

            # 1.5. merge and train
            """ 
            while ∃ meaningful cluster-pair merge do
            论文中是迭代聚合直到没有新簇产生，这里却是直接指定迭代聚合次数
            merge XXX times
            """
            for i in range(1, config_dict["merge_iters"]+1):
                print('Iteration number {}'.format(i))
                logging.info('Iteration number {}'.format(i))

                # Entities
                """
                E_t <- UpdateJointFeatures(V_t)
                S_E <- TrainMentionPairScorer(E_t; G)
                E_t <- MergeClusters(S_E; E_t)
                """
                print('Train entity model and merge entity clusters...')
                logging.info('Train entity model and merge entity clusters...')
                train_and_merge(clusters=entity_clusters, other_clusters=event_clusters,
                                model=cd_entity_model, optimizer=cd_entity_optimizer,
                                loss=cd_entity_loss,device=device,topic=topic,is_event=False,epoch=epoch,
                                topics_counter=topics_counter, topics_num=topics_num,
                                threshold=entity_th)
                # Events
                """
                V_t <- UpdateJointFeatures(E_t)
                S_V <- TrainMentionPairScorer(V_t; G)
                V_t <- MergeClusters(S_V; V_t)
                """
                print('Train event model and merge event clusters...')
                logging.info('Train event model and merge event clusters...')
                train_and_merge(clusters=event_clusters, other_clusters=entity_clusters,
                                model=cd_event_model, optimizer=cd_event_optimizer,
                                loss=cd_event_loss,device=device,topic=topic,is_event=True,epoch=epoch,
                                topics_counter=topics_counter, topics_num=topics_num,
                                threshold=event_th)

        # 2. testing models on whole dev set once (one epoch)
        print('Testing models on dev set...')
        logging.info('Testing models on dev set...')

        threshold_list = config_dict["dev_th_range"]
        improved = False
        best_event_f1_for_th = 0
        best_entity_f1_for_th = 0

        if event_best_dev_f1 > 0:
            best_saved_cd_event_model = load_check_point(os.path.join(args.out_dir,
                                                                      'cd_event_best_model'))
            best_saved_cd_event_model.to(device)
        else:
            best_saved_cd_event_model = cd_event_model

        if entity_best_dev_f1 > 0:
            best_saved_cd_entity_model = load_check_point(os.path.join(args.out_dir,
                                                                       'cd_entity_best_model'))
            best_saved_cd_entity_model.to(device)
        else:
            best_saved_cd_entity_model = cd_entity_model

        for event_threshold in threshold_list:
            for entity_threshold in threshold_list:
                config_dict["event_merge_threshold"] = event_threshold
                config_dict["entity_merge_threshold"] = entity_threshold
                print('Testing models on dev set with threshold={}'.format((event_threshold,entity_threshold)))
                logging.info('Testing models on dev set with threshold={}'.format((event_threshold,entity_threshold)))

                # test event coref on dev
                event_f1, _ = test_models(dev_set, cd_event_model, best_saved_cd_entity_model, device,
                                                  config_dict, write_clusters=False, out_dir=args.out_dir,
                                                  doc_to_entity_mentions=doc_to_entity_mentions, analyze_scores=False)

                # test entity coref on dev
                _, entity_f1 = test_models(dev_set, best_saved_cd_event_model, cd_entity_model, device,
                                                  config_dict, write_clusters=False, out_dir=args.out_dir,
                                                  doc_to_entity_mentions=doc_to_entity_mentions, analyze_scores=False)

                if event_f1 > best_event_f1_for_th:
                    best_event_f1_for_th = event_f1
                    best_event_th = (event_threshold,entity_threshold)

                if entity_f1 > best_entity_f1_for_th:
                    best_entity_f1_for_th = entity_f1
                    best_entity_th = (event_threshold,entity_threshold)

        event_f1 = best_event_f1_for_th
        entity_f1 = best_entity_f1_for_th
        save_epoch_f1(event_f1, entity_f1, epoch, best_event_th, best_entity_th)

        config_dict["event_merge_threshold"] = orig_event_th
        config_dict["entity_merge_threshold"] = orig_entity_th

        if event_f1 > event_best_dev_f1:
            event_best_dev_f1 = event_f1
            best_event_epoch = epoch
            save_check_point(cd_event_model, os.path.join(args.out_dir, 'cd_event_best_model'))
            improved = True
            patient_counter = 0
        if entity_f1 > entity_best_dev_f1:
            entity_best_dev_f1 = entity_f1
            best_entity_epoch = epoch
            save_check_point(cd_entity_model, os.path.join(args.out_dir, 'cd_entity_best_model'))
            improved = True
            patient_counter = 0

        if not improved:
            patient_counter += 1

        save_training_checkpoint(epoch, cd_event_model, cd_event_optimizer, event_best_dev_f1,
                                 filename=os.path.join(args.out_dir, 'cd_event_model_state'))
        save_training_checkpoint(epoch, cd_entity_model, cd_entity_optimizer, entity_best_dev_f1,
                                 filename=os.path.join(args.out_dir, 'cd_entity_model_state'))

        if patient_counter >= config_dict["patient"]:
            logging.info('Early Stopping!')
            print('Early Stopping!')
            save_summary(event_best_dev_f1, entity_best_dev_f1, best_event_epoch, best_entity_epoch, epoch)
            break


def train_and_merge(clusters, other_clusters, model, optimizer,
                    loss, device, topic, is_event, epoch,
                    topics_counter, topics_num, threshold):
    '''
    This function trains event/entity and then uses agglomerative clustering algorithm that
    merges event/entity clusters

    :param clusters: current event/entity clusters
    :param other_clusters: should be the event current clusters if clusters = entity clusters
    and vice versa.
    :param model: event/entity model (according to clusters parameter)
    :param optimizer: event/entity optimizer (according to clusters parameter)
    :param loss: event/entity loss (according to clusters parameter)
    :param device: gpu/cpu Pytorch device
    :param topic: Topic object represents the current topic
    :param is_event: whether to currently handle event mentions or entity mentions
    :param epoch: current epoch number
    :param topics_counter: the number of current topic
    :param topics_num: total number of topics
    :param threshold: merging threshold
    :return:
    '''

    # 根据事件(实体)共指更新实体(事件)表征
    #   其实只更新每个实体(事件)指称的d_m，而s_m和c_m是固定的。
    #   Update arguments/predicates vectors according to the other clusters state
    update_args_feature_vectors(clusters, other_clusters, model, device, is_event)

    # 训练model
    #   生成数据
    cluster_pairs, test_cluster_pairs = generate_cluster_pairs(clusters, is_train=True)
    #   Train pairwise event/entity coreference scorer
    train(cluster_pairs, model, optimizer, loss,
          device, topic.docs, epoch, topics_counter, topics_num, config_dict, is_event,
          other_clusters)

    with torch.no_grad():
        update_lexical_vectors(clusters, model, device, is_event, requires_grad=False)

        event_mentions, entity_mentions = topic_to_mention_list(topic, is_gold=True)

        # Update span representations after training
        create_mention_span_representations(event_mentions, model, device, topic.docs,
                                            is_event=True, requires_grad=False)
        create_mention_span_representations(entity_mentions, model, device, topic.docs,
                                            is_event=False, requires_grad=False)

        cluster_pairs = test_cluster_pairs

        # Merge clusters till reaching the threshold
        merge(clusters, cluster_pairs, other_clusters, model, device, topic.docs, epoch,
              topics_counter, topics_num, threshold, is_event,
              config_dict["use_args_feats"], config_dict["use_binary_feats"])


def save_epoch_f1(event_f1, entity_f1, epoch,  best_event_th, best_entity_th):
    '''
    Write to a text file B-cubed F1 measures of both event and entity clustering
    according to the models' predictions on the dev set after each training epoch.
    :param event_f1: B-cubed F1 measure for event coreference
    :param entity_f1: B-cubed F1 measure for entity coreference
    :param epoch: current epoch number
    :param best_event_th: best found merging threshold for event coreference
    :param best_entity_th: best found merging threshold for event coreference
    '''
    f = open(os.path.join(args.out_dir,'epochs_scores.txt'),'a')
    f.write('Epoch {} -  Event F1: {:.3f} with th = {}  Entity F1: {:.3f} with th = {}  \n'.format(epoch,event_f1,best_event_th, entity_f1, best_entity_th))
    f.close()


def save_summary(best_event_score,best_entity_score, best_event_epoch,best_entity_epoch, total_epochs):
    '''
    Writes to a file a summary of the training (best scores, their epochs, and total number of
    epochs)
    :param best_event_score: best event coreference score on the dev set
    :param best_entity_score: best entity coreference score on the dev set
    :param best_event_epoch: the epoch of the best event coreference
    :param best_entity_epoch: the epoch of the best entity coreference
    :param total_epochs: total number of epochs
    '''
    f = open(os.path.join(args.out_dir, 'summary.txt'), 'w')
    f.write('Best Event F1: {:.3f} epoch: {} \n Best Entity F1: {:.3f} epoch: '
            '{} \n Training epochs: {}'.format(best_event_score,best_event_epoch,best_entity_score
                                               ,best_entity_epoch, total_epochs))


def save_training_checkpoint(epoch, model, optimizer, best_f1, filename):
    '''
    Saves model's checkpoint after each epoch
    :param epoch: epoch number
    :param model: the model to save
    :param optimizer: Pytorch optimizer
    :param best_f1: the best B-cubed F1 score so far
    :param filename: the filename of the checkpoint file
    '''
    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'best_f1': best_f1 }
    torch.save(state, filename)


def load_training_checkpoint(model, optimizer, filename, device):
    '''
    Loads checkpoint from a file
    :param model: an initialized model (CDCorefScorer)
    :param optimizer: new Pytorch optimizer
    :param filename: the checkpoint filename
    :param device: gpu/cpu device
    :return: model, optimizer, epoch, best_f1 loaded from the checkpoint.
    '''
    print("Loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_f1 = checkpoint['best_f1']
    print("Loaded checkpoint '{}' (epoch {})"
                .format(filename, checkpoint['epoch']))

    model = model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    return model, optimizer, start_epoch, best_f1


def load_train_set_and_dev_set(train_path: str, dev_path: str) -> Tuple[Corpus, Corpus]:
    """
    Read pkl file of training set and dev set, return Corpus objs.

    :param train_path: The path to a strining set pkl file.
        e.g. 'data/processed/cybulska_setup/full_swirl_ecb/training_data'
    :param dev_path: The path to a dev set pkl file.
        e.g. 'data/processed/cybulska_setup/full_swirl_ecb/dev_data'
    :return: Corpus obj of training set and dev set.
    """
    logging.info('Loading training data...')
    with open(train_path, 'rb') as f:
        training_data: Corpus = cPickle.load(f)  # src.shared.classes.Corpus object
    logging.info('Loading training and dev data...')
    with open(dev_path, 'rb') as f:
        dev_data: Corpus = cPickle.load(f)  # src.shared.classes.Corpus object
    logging.info('Training and dev data have been loaded.')
    return training_data, dev_data


def main():
    '''
    This script loads the train and dev sets, initializes models, optimizers and loss functions,
    then, it runs the training procedure that alternates between entity and event training and
    their clustering.
    After each epoch, it runs the inference procedure on the dev set, calculates
    the B-cubed measure and use it to tune the model and its hyper-parameters.
    Finally, it saves the entity and event models that achieved the best B-cubed scores
    on the dev set.
    '''
    (training_data, dev_data) = load_train_set_and_dev_set(config_dict["train_path"], config_dict["dev_path"])
    train_model(training_data, dev_data)


if __name__ == '__main__':
    main()
