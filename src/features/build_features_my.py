# 标准库
import os
import sys
import json
import torch
import argparse
import _pickle as cPickle
from typing import Dict, List, Tuple, Union  # for type hinting
from collections import defaultdict

# 三方库
import spacy
from nltk.corpus import wordnet as wn
from breakpointAlarm import alarm

# 本地库
from src.shared.classes import Document, Sentence, Token, EventMention, EntityMention
from src.features.swirl_parsing import parse_swirl_output
from src.features.allen_srl_reader import read_srl
from src.features.create_elmo_embeddings import *
from src.features.extraction_utils import *
# for pack in os.listdir("src"):
#     sys.path.append(os.path.join("src", pack))
# sys.path.append("/src/shared/")

# spaCy使用了预训练模型“en”
nlp = spacy.load('en')

# parse the arguments in command
parser = argparse.ArgumentParser(description='Feature extraction (predicate-argument structures,'
                                             'mention heads, and ELMo embeddings)')
parser.add_argument('--config_path', type=str,
                    help=' The path to the configuration json file')
parser.add_argument('--output_path', type=str,
                    help=' The path to output folder (Where to save the processed data)')
args = parser.parse_args()

# make the output dir
out_dir = args.output_path
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# load config file
with open(args.config_path, 'r') as js_file:
    config_dict = json.load(js_file)
# copy config file into output dir
with open(os.path.join(args.output_path, 'build_features_config.json'), "w") as js_file:
    json.dump(config_dict, js_file, indent=4, sort_keys=True)

# 配置logging
import logging
logging.basicConfig(
    # 使用fileHandler,日志文件在输出路径中(test_log.txt)
    filename=os.path.join(out_dir, "test_log.txt"),
    filemode="w",
    # 配置日志级别
    level=logging.INFO
)


def load_mentions_from_json(mentions_json_file: str,
                            docs: Dict[str, Document],
                            is_event: bool, is_gold_mentions) -> None:
    """
    This function extract mention info from a given json file and add the
    mention info to param *docs*. The *docs* param has a structure as shown below::

        普通变量是docs本来就有的信息，
        尖括号中的变量是本函数运行后添加的信息。
        Document_obj
            Document_obj.sentences -> Sentence_obj
            <gold/pred_event/entity_mentions> -> Mention obj
        Sentence_obj
            Sentence_obj.tokens -> Token_obj
        <Mention_obj>
            <cd/wd_coref_chain> -> Coref_chain_str
            <doc_id> -> Document_obj
            <sent_id> -> Sentence_obj
            <tokens> -> Token_obj
        Token_obj
            <gold_event/entity_cd/wd_coref_chain> -> Coref_chain_str

    * This function has a Sanity check. Check whether mention in json file and the corresponding
      mention in docs has same token. If not this function stop and err info will be printed.

    * The json file is like (a list of **mention dict**)::

        [
            {
                "coref_chain": "HUM16284637796168708",
                "doc_id": "1_10ecb",
                "is_continuous": true,
                "is_singleton": false,
                "mention_type": "HUM",
                "score": -1.0,
                "sent_id": 0,
                "tokens_number": [
                    13
                ],
                "tokens_str": "rep"
            },
            {
                "coref_chain": "HUM16236184328979740",
                "doc_id": "1_10ecb",
                "is_continuous": true,
                "is_singleton": false,
                "mention_type": "HUM",
                "score": -1.0,
                "sent_id": 0,
                "tokens_number": [
                    3,
                    4
                ],
                "tokens_str": "Tara Reid"
            },


    :param mentions_json_file: path to the JSON file that contains the mentions.
        The json file has a content like that of
        ECB_Dev/Test/Train_Entity/Event_gold_mentions.json.

    :param docs: { 'XX_XXecb': a src.shared.classes.Document Obj }

    :param is_event: a boolean indicates whether the mention in json file is event or entity
     mentions.

    :param is_gold_mentions: a boolean indicates whether the mention in json file is gold or
     predicted mentions.
    """

    with open(mentions_json_file, 'r') as js_file:
        js_mentions = json.load(js_file)

    for js_mention in js_mentions:
        doc_id = js_mention["doc_id"].replace('.xml', '')  # 这是废话，因为doc_id里都没有‘.xml’
        sent_id = js_mention["sent_id"]
        tokens_numbers = js_mention["tokens_number"]
        mention_type = js_mention["mention_type"]
        is_singleton = js_mention["is_singleton"]
        is_continuous = js_mention["is_continuous"]
        score = js_mention["score"]
        mention_str = js_mention["tokens_str"]
        if mention_str is None:
            print('Err: mention str is None:', js_mention)
        coref_chain = js_mention["coref_chain"]
        head_text, head_lemma = find_head(mention_str)

        """Sanity check
        Check whether mention in json file and the corresponding Mention obj in 
        docs has same tokens. 
        """
        # Find the tokens of corresponding mention in docs.
        try:
            token_objects = docs[doc_id].get_sentences()[sent_id].find_mention_tokens(tokens_numbers)
        except:
            print('error when looking for mention tokens')
            print('doc id {} sent id {}'.format(doc_id, sent_id))
            print('token numbers - {}'.format(str(tokens_numbers)))
            print('mention string {}'.format(mention_str))
            print('sentence - {}'.format(docs[doc_id].get_sentences()[sent_id].get_raw_sentence()))
            raise  # stop the script
        if not token_objects:
            # Never hit this if condition. token_objects = []?.
            print('Can not find tokens of a mention - {} {} {}'.format(doc_id, sent_id,tokens_numbers))
        # whether the tokens are same
            pass

        # 1. add coref chain to Token.
        if is_gold_mentions:
            for token in token_objects:
                if is_event:
                    token.gold_event_coref_chain.append(coref_chain)
                else:
                    token.gold_entity_coref_chain.append(coref_chain)

        # 2. Create Mention
        if is_event:
            mention = EventMention(doc_id, sent_id, tokens_numbers, token_objects, mention_str, head_text,
                                   head_lemma, is_singleton, is_continuous, coref_chain)
        else:
            mention = EntityMention(doc_id, sent_id, tokens_numbers, token_objects, mention_str, head_text,
                                    head_lemma, is_singleton, is_continuous, coref_chain, mention_type)
        mention.probability = score
        # a confidence score for predicted mentions (if used), gold mentions prob is setted to 1.0 in the json file.

        # 3. add Mention to Sentence
        if is_gold_mentions:
            docs[doc_id].get_sentences()[sent_id].add_gold_mention(mention, is_event)
        else:
            docs[doc_id].get_sentences()[sent_id].add_predicted_mention(
                mention, is_event,
                relaxed_match=config_dict["relaxed_match_with_gold_mention"])


def load_gold_mentions(docs: Dict[str, Document], events_json: str, entities_json: str) -> None:
    """
    This function loads given event and entity mentions as gold mention.
    No return. This function add the mention info into *docs*, instead of output
    a return value.

    Example of *docs*::
        {'XX_XXecb': Document Obj, ... }

    Example of *events_json* and *entities_json*::
        "data/interim/cybulska_setup/ECB_Train_Event_gold_mentions.json"

    :param docs: A dict of Document objects of train/text/dev set.
    :param events_json:  Path to the JSON file which contains the gold event
    mentions of a specific split - train/dev/test
    :param entities_json: Path to the JSON file which contains the gold entity
    mentions of a specific split - train/dev/test
    """
    load_mentions_from_json(events_json, docs, is_event=True, is_gold_mentions=True)
    load_mentions_from_json(entities_json, docs, is_event=False, is_gold_mentions=True)


def load_predicted_mentions(docs: Dict[str, Document], events_json: str, entities_json: str) -> None:
    """
    This function loads given event and entity mentions as predicted mention.
    No return. This function add the mention info into *docs*, instead of output
    a return value.

    Example of *docs*::
        {'XX_XXecb': Document Obj, ... }

    Example of *events_json* and *entities_json*::
        "data/interim/cybulska_setup/ECB_Train_Event_pred_mentions.json"

    :param docs: A dict of document objects of train/text/dev set.
    :param events_json:  Path to the JSON file which contains the predicted event
    mentions of a specific split - train/dev/test
    :param entities_json: Path to the JSON file which contains the predicted entity
    mentions of a specific split - train/dev/test
    """
    load_mentions_from_json(events_json, docs, is_event=True, is_gold_mentions=False)
    load_mentions_from_json(entities_json, docs, is_event=False, is_gold_mentions=False)


# def load_gold_data(split_txt_file, events_json, entities_json):
#     '''
#     This function loads the texts of each split and its gold mentions, create document objects
#     and stored the gold mentions within their suitable document objects
#
#     :param split_txt_file: the text file of each split is written as 5 columns (stored in data/intermid)
#     :param events_json: a JSON file contains the gold event mentions
#     :param entities_json: a JSON file contains the gold event mentions
#     :return:
#     '''
#     logging.info('Loading gold mentions...')
#     docs = load_ECB_plus(split_txt_file)
#     load_gold_mentions(docs, events_json, entities_json)
#     return docs


def load_predicted_data(docs: dict, pred_events_json: str, pred_entities_json: str):
    '''
    This function loads the predicted mentions and stored them within their
    suitable document objects (suitable for loading the test data)

    :param docs: dictionary that contains the document objects
    :param pred_events_json: path to the JSON file contains predicted event mentions
    :param pred_entities_json: path to the JSON file contains predicted entities mentions
    :return:
    '''
    logging.info('Loading predicted mentions...')
    load_predicted_mentions(docs, pred_events_json, pred_entities_json)


def find_head(mention_str: str) -> Tuple[str, str]:
    """
    This function find the root in dependency parsing of param *mention_str*.

    The head of the root is itself. The dependency type of the root is 'ROOT'.
    Based on those feature, we can find the root. For example::
        >>> import spacy
        >>> nlp = spacy.load('en')
        >>> text = "The yellow dog eat shite."
        >>> doc = nlp(text)
        >>> [(i, i.head) for i in doc]
        [(The, dog), (yellow, dog), (dog, eat), (eat, eat), (shite, eat), (., eat)]

    Usually, a mention or a sentence has only one root, as the example above shows.
    However, if your *mention_str* is long and complex, there can be more roots. For example::
        >>> text = "The yellow dog eat shite, but the white cat eat fish."
        >>> doc = nlp(text)
        >>> [(i, i.head) for i in doc]
        [(The, dog), (yellow, dog), (dog, eat), (eat, eat), (shite, eat), (,, eat), (but, eat), (the, cat), (white, cat), (cat, eat), (eat, eat), (fish, eat), (., eat)]
        >>> [(i, i.dep_) for i in doc]
        [(The, 'det'), (yellow, 'amod'), (dog, 'nsubj'), (eat, 'ROOT'), (shite, 'dobj'), (,, 'punct'), (but, 'cc'), (the, 'det'), (white, 'amod'), (cat, 'nsubj'), (eat, 'conj'), (fish, 'dobj'), (., 'punct')]
    The two 'eat' are root.
    In this case, this function find only the fist root.
    But, this function is not designed for this case. The param *mention_str* should be a real shour mention string which has only one root.

    After find the root, this function returns (text_of_root, lemma_of_root).
    Specially, lemma of pronone is '-PRON-', for example the 'you' in 'spaCy is designed to help you do real work'.
    In this case, this function returns (text_of_root, lower_case_of_root)

    :param mention_str: A mention string.
    :return: (text_of_root, lemma_of_root) or (text_of_root, lower_case_of_root)
    """
    mention = nlp(mention_str)
    for token in mention:
        # The token whose head is itself is the root
        if token.head == token:  # if token.dep_ == 'ROOT'
            if token.lemma_ == u'-PRON-':
                return token.text, token.text.lower()
            return token.text, token.lemma_


def have_string_match(mention,arg_str ,arg_start, arg_end):
    '''
    This function checks whether a given entity mention has a string match (strict or relaxed)
    with a span of an extracted argument
    :param mention: a candidate entity mention
    :param arg_str: the argument's text
    :param arg_start: the start index of the argument's span
    :param arg_end: the end index of the argument's span
    :return: True if there is a string match (strict or relaxed) between the entity mention
    and the extracted argument's span, and false otherwise
    '''
    if mention.mention_str == arg_str and mention.start_offset == arg_start:  # exact string match + same start index
        return True
    if mention.mention_str == arg_str:  # exact string match
        return True
    if mention.start_offset >= arg_start and mention.end_offset <= arg_end:  # the argument span contains the mention span
        return True
    if arg_start >= mention.start_offset and arg_end <= mention.end_offset:  # the mention span contains the mention span
        return True
    if len(set(mention.tokens_numbers).intersection(set(range(arg_start,arg_end + 1)))) > 0: # intersection between the mention's tokens and the argument's tokens
        return True
    return False


def add_arg_to_event(entity, event, rel_name):
    '''
    Adds the entity mention as an argument (in a specific role) of an event mention and also adds the
    event mention as predicate (in a specific role) of the entity mention
    :param entity: an entity mention object
    :param event: an event mention object
    :param rel_name: the specific role
    '''
    if rel_name == 'A0':
        event.arg0 = (entity.mention_str, entity.mention_id)
        entity.add_predicate((event.mention_str, event.mention_id), 'A0')
    elif rel_name == 'A1':
        event.arg1 = (entity.mention_str, entity.mention_id)
        entity.add_predicate((event.mention_str, event.mention_id), 'A1')
    elif rel_name == 'AM-TMP':
        event.amtmp = (entity.mention_str, entity.mention_id)
        entity.add_predicate((event.mention_str, event.mention_id), 'AM-TMP')
    elif rel_name == 'AM-LOC':
        event.amloc = (entity.mention_str, entity.mention_id)
        entity.add_predicate((event.mention_str, event.mention_id), 'AM-LOC')


def find_argument(rel_name, rel_tokens, matched_event, sent_entities, sent_obj, is_gold, srl_obj):
    '''
    This function matches between an argument of an event mention and an entity mention.
    :param rel_name: the specific role of the argument
    :param rel_tokens: the argument's tokens
    :param matched_event: the event mention
    :param sent_entities: a entity mentions exist in the event's sentence.
    :param sent_obj: the object represents the sentence
    :param is_gold: whether the argument need to be matched with a gold mention or not
    :param srl_obj: an object represents the extracted SRL argument.
    :return True if the extracted SRL argument was matched with an entity mention.
    '''
    arg_start_ix = rel_tokens[0]
    if len(rel_tokens) > 1:
        arg_end_ix = rel_tokens[1]
    else:
        arg_end_ix = rel_tokens[0]

    if arg_end_ix >= len(sent_obj.get_tokens()):
        print('argument bound mismatch with sentence length')
        print('arg start index - {}'.format(arg_start_ix))
        print('arg end index - {}'.format(arg_end_ix))
        print('sentence length - {}'.format(len(sent_obj.get_tokens())))
        print('raw sentence: {}'.format(sent_obj.get_raw_sentence()))
        print('matched event: {}'.format(str(matched_event)))
        print('srl obj - {}'.format(str(srl_obj)))

    arg_str, arg_tokens = sent_obj.fetch_mention_string(arg_start_ix, arg_end_ix)

    entity_found = False
    matched_entity = None
    for entity in sent_entities:
        if have_string_match(entity, arg_str, arg_start_ix, arg_end_ix):
            if rel_name == 'AM-TMP' and entity.mention_type != 'TIM':
                continue
            if rel_name == 'AM-LOC' and entity.mention_type != 'LOC':
                continue
            entity_found = True
            matched_entity = entity
            break
    if entity_found:
        add_arg_to_event(matched_entity, matched_event, rel_name)
        if is_gold:
            return True
        else:
            if matched_entity.gold_mention_id is not None:
                return True
            else:
                return False
    else:
        return False


def match_allen_srl_structures(dataset, srl_data, is_gold):
    '''
    Matches between extracted predicates and event mentions and between their arguments and
    entity mentions, designed to handle the output of Allen NLP SRL system
    :param dataset: an object represents the spilt (train/dev/test)
    :param srl_data: a dictionary contains the predicate-argument structures
    :param is_gold: whether to match predicate-argument structures with gold mentions or with predicted mentions
    '''
    matched_events_count = 0
    matched_args_count = 0

    for topic_id, topic in dataset.topics.items():
        for doc_id, doc in topic.docs.items():
            for sent_id, sent in doc.get_sentences().items():
                # Handling nominalizations in case we don't use syntactic dependencies (which already handle this)
                if not config_dict["use_dep"]:
                    sent_str = sent.get_raw_sentence()
                    parsed_sent = nlp(sent_str)
                    find_nominalizations_args(parsed_sent, sent, is_gold)
                sent_srl_info = None

                if doc_id in srl_data:
                    doc_srl = srl_data[doc_id]
                    if int(sent_id) in doc_srl:
                        sent_srl_info = doc_srl[int(sent_id)]

                if sent_srl_info is not None:
                    for event_srl in sent_srl_info.srl:
                        event_text = event_srl.verb.text
                        event_ecb_tok_ids = event_srl.verb.ecb_tok_ids

                        if is_gold:
                            sent_events = sent.gold_event_mentions
                            sent_entities = sent.gold_entity_mentions
                        else:
                            sent_events = sent.pred_event_mentions
                            sent_entities = sent.pred_entity_mentions
                        event_found = False
                        matched_event = None

                        for event_mention in sent_events:
                            if event_ecb_tok_ids == event_mention.tokens_numbers or \
                                    event_text == event_mention.mention_str or \
                                    event_text in event_mention.mention_str or \
                                    event_mention.mention_str in event_text:
                                event_found = True
                                matched_event = event_mention
                                if is_gold:
                                    matched_events_count += 1
                                elif matched_event.gold_mention_id is not None:
                                    matched_events_count += 1
                            if event_found:
                                break
                        if event_found:
                            if event_srl.arg0 is not None:
                                if match_entity_with_srl_argument(sent_entities, matched_event,
                                                                  event_srl.arg0, 'A0', is_gold):
                                    matched_args_count += 1

                            if event_srl.arg1 is not None:
                                if match_entity_with_srl_argument(sent_entities, matched_event,
                                                                  event_srl.arg1, 'A1', is_gold):
                                    matched_args_count += 1
                            if event_srl.arg_tmp is not None:
                                if match_entity_with_srl_argument(sent_entities, matched_event,
                                                                  event_srl.arg_tmp, 'AM-TMP', is_gold):
                                    matched_args_count += 1

                            if event_srl.arg_loc is not None:
                                if match_entity_with_srl_argument(sent_entities, matched_event,
                                                                  event_srl.arg_loc, 'AM-LOC', is_gold):
                                    matched_args_count += 1

    logging.info('SRL matched events - ' + str(matched_events_count))
    logging.info('SRL matched args - ' + str(matched_args_count))


def match_entity_with_srl_argument(sent_entities, matched_event ,srl_arg,rel_name, is_gold):
    '''
    This function matches between an argument of an event mention and an entity mention.
    Designed to handle the output of Allen NLP SRL system
    :param sent_entities: the entity mentions in the event's sentence
    :param matched_event: the event mention
    :param srl_arg: the extracted argument
    :param rel_name: the role name
    :param is_gold: whether to match the argument with gold entity mention or with predicted entity mention
    :return:
    '''
    found_entity = False
    matched_entity = None
    for entity in sent_entities:
        if srl_arg.ecb_tok_ids == entity.tokens_numbers or \
                srl_arg.text == entity.mention_str or \
                srl_arg.text in entity.mention_str or \
                entity.mention_str in srl_arg.text:
            if rel_name == 'AM-TMP' and entity.mention_type != 'TIM':
                continue
            if rel_name == 'AM-LOC' and entity.mention_type != 'LOC':
                continue
            found_entity = True
            matched_entity = entity

        if found_entity:
            break

    if found_entity:
        add_arg_to_event(matched_entity, matched_event, rel_name)
        if is_gold:
            return True
        else:
            if matched_entity.gold_mention_id is not None:
                return True
            else:
                return False
    else:
        return False


def load_srl_info(dataset, srl_data, is_gold):
    '''
    Matches between extracted predicates and event mentions and between their arguments and
    entity mentions.
    :param dataset: an object represents the spilt (train/dev/test)
    :param srl_data: a dictionary contains the predicate-argument structures
    :param is_gold: whether to match predicate-argument structures with gold mentions or with predicted mentions
    '''
    matched_events_count = 0
    unmatched_event_count = 0
    matched_args_count = 0

    matched_identified_events = 0
    matched_identified_args = 0
    for topic_id, topic in dataset.topics.items():
        for doc_id, doc in topic.docs.items():
            for sent_id, sent in doc.get_sentences().items():
                # Handling nominalizations if we don't use dependency parsing (that already handles it)
                if not config_dict["use_dep"]:
                    sent_str = sent.get_raw_sentence()
                    parsed_sent = nlp(sent_str)
                    find_nominalizations_args(parsed_sent, sent, is_gold)
                sent_srl_info = {}

                if doc_id in srl_data:
                    doc_srl = srl_data[doc_id]
                    if int(sent_id) in doc_srl:
                        sent_srl_info = doc_srl[int(sent_id)]
                else:
                    print('doc not in srl data - ' + doc_id)

                for event_key, srl_obj in sent_srl_info.items():
                    if is_gold:
                        sent_events = sent.gold_event_mentions
                        sent_entities = sent.gold_entity_mentions
                    else:
                        sent_events = sent.pred_event_mentions
                        sent_entities = sent.pred_entity_mentions
                    event_found = False
                    matched_event = None
                    for event_mention in sent_events:
                        if event_key in event_mention.tokens_numbers:
                            event_found = True
                            matched_event = event_mention
                            if is_gold:
                                matched_events_count += 1
                            elif matched_event.gold_mention_id is not None:
                                    matched_events_count += 1
                        if event_found:
                            break
                    if event_found:
                        for rel_name, rel_tokens in srl_obj.arg_info.items():
                            if find_argument(rel_name, rel_tokens, matched_event, sent_entities, sent, is_gold,srl_obj):
                                matched_args_count += 1
                    else:
                        unmatched_event_count += 1
    logging.info('SRL matched events - ' + str(matched_events_count))
    logging.info('SRL unmatched events - ' + str(unmatched_event_count))
    logging.info('SRL matched args - ' + str(matched_args_count))


def find_topic_gold_clusters(topic):
    '''
    Finds the gold clusters of a specific topic
    :param topic: a topic object
    :return: a mapping of coref chain to gold cluster (for a specific topic) and the topic's mentions
    '''
    event_mentions = []
    entity_mentions = []
    # event_gold_tag_to_cluster = defaultdict(list)
    # entity_gold_tag_to_cluster = defaultdict(list)

    event_gold_tag_to_cluster = {}
    entity_gold_tag_to_cluster = {}

    for doc_id, doc in topic.docs.items():
        for sent_id, sent in doc.sentences.items():
            event_mentions.extend(sent.gold_event_mentions)
            entity_mentions.extend(sent.gold_entity_mentions)

    for event in event_mentions:
        if event.gold_tag != '-':
            if event.gold_tag not in event_gold_tag_to_cluster:
                event_gold_tag_to_cluster[event.gold_tag] = []
            event_gold_tag_to_cluster[event.gold_tag].append(event)
    for entity in entity_mentions:
        if entity.gold_tag != '-':
            if entity.gold_tag not in entity_gold_tag_to_cluster:
                entity_gold_tag_to_cluster[entity.gold_tag] = []
            entity_gold_tag_to_cluster[entity.gold_tag].append(entity)

    return event_gold_tag_to_cluster, entity_gold_tag_to_cluster, event_mentions, entity_mentions


def write_dataset_statistics(split_name: str, dataset: dict, check_predicted):
    '''
    Prints the split statistics.

    :param split_name: the split name (a string)
    :param dataset: an object represents the split
    :param check_predicted: whether to print statistics of predicted mentions too
    '''
    docs_count = 0
    sent_count = 0
    event_mentions_count = 0
    entity_mentions_count = 0
    event_chains_count = 0
    entity_chains_count = 0
    topics_count = len(dataset.topics.keys())
    predicted_events_count = 0
    predicted_entities_count = 0
    matched_predicted_event_count = 0
    matched_predicted_entity_count = 0


    for topic_id, topic in dataset.topics.items():
        event_gold_tag_to_cluster, entity_gold_tag_to_cluster, \
        event_mentions, entity_mentions = find_topic_gold_clusters(topic)

        docs_count += len(topic.docs.keys())
        sent_count += sum([len(doc.sentences.keys()) for doc_id, doc in topic.docs.items()])
        event_mentions_count += len(event_mentions)
        entity_mentions_count += len(entity_mentions)

        entity_chains = set()
        event_chains = set()

        for mention in entity_mentions:
            entity_chains.add(mention.gold_tag)

        for mention in event_mentions:
            event_chains.add(mention.gold_tag)

        # event_chains_count += len(set(event_gold_tag_to_cluster.keys()))
        # entity_chains_count += len(set(entity_gold_tag_to_cluster.keys()))

        event_chains_count += len(event_chains)
        entity_chains_count += len(entity_chains)

        if check_predicted:
            for doc_id, doc in topic.docs.items():
                for sent_id, sent in doc.sentences.items():
                    pred_events = sent.pred_event_mentions
                    pred_entities = sent.pred_entity_mentions

                    predicted_events_count += len(pred_events)
                    predicted_entities_count += len(pred_entities)

                    for pred_event in pred_events:
                        if pred_event.has_compatible_mention:
                            matched_predicted_event_count += 1

                    for pred_entity in pred_entities:
                        if pred_entity.has_compatible_mention:
                            matched_predicted_entity_count += 1

    with open(os.path.join(args.output_path, '{}_statistics.txt'.format(split_name)), 'w') as f:
        f.write('Number of topics - {}\n'.format(topics_count))
        f.write('Number of documents - {}\n'.format(docs_count))
        f.write('Number of sentences - {}\n'.format(sent_count))
        f.write('Number of event mentions - {}\n'.format(event_mentions_count))
        f.write('Number of entity mentions - {}\n'.format(entity_mentions_count))

        if check_predicted:
            f.write('Number of predicted event mentions  - {}\n'.format(predicted_events_count))
            f.write('Number of predicted entity mentions - {}\n'.format(predicted_entities_count))
            f.write('Number of predicted event mentions that match gold mentions- '
                    '{} ({}%)\n'.format(matched_predicted_event_count,
                                        (matched_predicted_event_count/float(event_mentions_count)) *100 ))
            f.write('Number of predicted entity mentions that match gold mentions- '
                    '{} ({}%)\n'.format(matched_predicted_entity_count,
                                        (matched_predicted_entity_count / float(entity_mentions_count)) * 100))


def obj_dict(obj):
    obj_d = obj.__dict__
    obj_d = stringify_keys(obj_d)
    return obj_d


def stringify_keys(d):
    """Convert a dict's keys to strings if they are not."""
    for key in d.keys():

        # check inner dict
        if isinstance(d[key], dict):
            value = stringify_keys(d[key])
        else:
            value = d[key]

        # convert nonstring to string if needed
        if not isinstance(key, str):
            try:
                d[str(key)] = value
            except Exception:
                try:
                    d[repr(key)] = value
                except Exception:
                    pass

            # delete old key
            del d[key]
    return d


def set_elmo_embed_to_mention(mention, sent_embeddings):
    '''
    Sets the ELMo embeddings of a mention
    :param mention: event/entity mention object
    :param sent_embeddings: the embedding for each word in the sentence produced by ELMo model
    :return:
    '''
    head_index = mention.get_head_index()
    head_embeddings = sent_embeddings[int(head_index)]
    mention.head_elmo_embeddings = torch.from_numpy(head_embeddings)


def set_elmo_embeddings_to_mentions(elmo_embedder, sentence, set_pred_mentions):
    '''
     Sets the ELMo embeddings for all the mentions in the sentence
    :param elmo_embedder: a wrapper object for ELMo model of Allen NLP
    :param sentence: a sentence object
    '''
    avg_sent_embeddings = elmo_embedder.get_elmo_avg(sentence)
    event_mentions = sentence.gold_event_mentions
    entity_mentions = sentence.gold_entity_mentions

    for event in event_mentions:
        set_elmo_embed_to_mention(event,avg_sent_embeddings)

    for entity in entity_mentions:
        set_elmo_embed_to_mention(entity, avg_sent_embeddings)

    # Set the contextualized vector also for predicted mentions
    if set_pred_mentions:
        event_mentions = sentence.pred_event_mentions
        entity_mentions = sentence.pred_entity_mentions

        for event in event_mentions:
            set_elmo_embed_to_mention(event, avg_sent_embeddings)  # set the head contextualized vector

        for entity in entity_mentions:
            set_elmo_embed_to_mention(entity, avg_sent_embeddings)  # set the head contextualized vector


def load_elmo_embeddings(dataset, elmo_embedder, set_pred_mentions):
    '''
    Sets the ELMo embeddings for all the mentions in the split
    :param dataset: an object represents a split (train/dev/test)
    :param elmo_embedder: a wrapper object for ELMo model of Allen NLP
    :return:
    '''
    for topic_id, topic in dataset.topics.items():
        for doc_id, doc in topic.docs.items():
            for sent_id, sent in doc.get_sentences().items():
                set_elmo_embeddings_to_mentions(elmo_embedder, sent, set_pred_mentions)


def main(args):
    """
        This script loads the train, dev and test json files (contain the gold entity and event
        mentions) builds mention objects, extracts predicate-argument structures, mention head
        and ELMo embeddings for each mention.

        Runs data processing scripts to turn intermediate data from (../intermid) into
        processed data ready to use in training and inference(saved in ../processed).
    """

    # 1. load and create Document, Sentence and Token objs.
    logging.info('Training data - loading tokens')
    training_data: Dict[str, Document] = load_ECB_plus(config_dict["train_text_file"])
    """{'XX_XXecb': Document Obj, ... } """
    logging.info('Dev data - Loading tokens')
    dev_data: Dict[str, Document] = load_ECB_plus(config_dict["dev_text_file"])
    """{'XX_XXecb': Document Obj, ... } """
    logging.info('Test data - Loading tokens')
    test_data: Dict[str, Document] = load_ECB_plus(config_dict["test_text_file"])
    """{'XX_XXecb': Document Obj, ... } """

    # 2. load and create gold Mention objs
    logging.info('Training data - Loading gold mentions')
    load_gold_mentions(training_data,
                       config_dict["train_event_mentions"], config_dict["train_entity_mentions"])
    logging.info('Dev data - Loading gold mentions')
    load_gold_mentions(dev_data,
                       config_dict["dev_event_mentions"], config_dict["dev_entity_mentions"])
    logging.info('Test data - Loading gold mentions')
    load_gold_mentions(test_data,
                       config_dict["test_event_mentions"], config_dict["test_entity_mentions"])

    # 3. load and create predicted Mention objs
    if config_dict["load_predicted_mentions"]:
        logging.info('Test data - Loading predicted mentions')
        load_predicted_mentions(test_data,
                                config_dict["pred_event_mentions"], config_dict["pred_entity_mentions"])

    # 4. create Corpus objs
    train_set = order_docs_by_topics(training_data)
    dev_set = order_docs_by_topics(dev_data)
    test_set = order_docs_by_topics(test_data)

    # 5. statistic number of t,d,s,em,vm in each split
    write_dataset_statistics('train', train_set, check_predicted=False)
    write_dataset_statistics('dev', dev_set, check_predicted=False)
    write_dataset_statistics('test', test_set, check_predicted=config_dict["load_predicted_mentions"])

    with open('output/train-base.pkl', 'wb') as f:
        cPickle.dump(train_set, f)
    with open('output/dev-base.pkl', 'wb') as f:
        cPickle.dump(dev_set, f)
    with open('output/test-base.pkl', 'wb') as f:
        cPickle.dump(test_set, f)

    with open('output/train-base.pkl', 'rb') as f:
        train_set = cPickle.load(f)
    with open('output/dev-base.pkl', 'rb') as f:
        dev_set = cPickle.load(f)
    with open('output/test-base.pkl', 'rb') as f:
        test_set = cPickle.load(f)

    # 6.load srl
    if config_dict["use_srl"]:
        logging.info('Loading SRL info')
        if config_dict["use_allen_srl"]:
            # use the SRL system which is implemented in AllenNLP (currently - a deep BiLSTM model (He et al, 2017).)
            srl_data = read_srl(config_dict["srl_output_path"])
            #
            logging.info('Training gold mentions - loading SRL info')
            match_allen_srl_structures(train_set, srl_data, is_gold=True)
            logging.info('Dev gold mentions - loading SRL info')
            match_allen_srl_structures(dev_set, srl_data, is_gold=True)
            logging.info('Test gold mentions - loading SRL info')
            match_allen_srl_structures(test_set, srl_data, is_gold=True)
            if config_dict["load_predicted_mentions"]:
                logging.info('Test predicted mentions - loading SRL info')
                match_allen_srl_structures(test_set, srl_data, is_gold=False)
        else:  # Use SwiRL SRL system (Surdeanu et al., 2007)
            srl_data = parse_swirl_output(config_dict["srl_output_path"])
            """
            example::
            
                {
                    '10_13ecbplus':{
                        0:{},
                        1:{},
                        2:{
                            2: src.shared.classes.Srl_info obj,
                            15: src.shared.classes.Srl_info obj,
                            ...
                        },
                        ...
                    },
                    ...
                }    
            """

            logging.info('Training gold mentions - loading SRL info')
            load_srl_info(train_set, srl_data, is_gold=True)
            logging.info('Dev gold mentions - loading SRL info')
            load_srl_info(dev_set, srl_data, is_gold=True)
            logging.info('Test gold mentions - loading SRL info')
            load_srl_info(test_set, srl_data, is_gold=True)
            if config_dict["load_predicted_mentions"]:
                logging.info('Test predicted mentions - loading SRL info')
                load_srl_info(test_set, srl_data, is_gold=False)

    # 7. load depprase
    if config_dict["use_dep"]:  # use dependency parsing
        logging.info('Augmenting predicate-arguments structures using dependency parser')
        logging.info('Training gold mentions - loading predicates and their arguments with dependency parser')
        find_args_by_dependency_parsing(train_set, is_gold=True)
        logging.info('Dev gold mentions - loading predicates and their arguments with dependency parser')
        find_args_by_dependency_parsing(dev_set, is_gold=True)
        logging.info('Test gold mentions - loading predicates and their arguments with dependency parser')
        find_args_by_dependency_parsing(test_set, is_gold=True)
        if config_dict["load_predicted_mentions"]:
            logging.info('Test predicted mentions - loading predicates and their arguments with dependency parser')
            find_args_by_dependency_parsing(test_set, is_gold=False)

    # 8. load left_right_mentiosn
    if config_dict["use_left_right_mentions"]:  # use left and right mentions heuristic
        logging.info('Augmenting predicate-arguments structures using leftmost and rightmost entity mentions')
        logging.info('Training gold mentions - loading predicates and their arguments ')
        find_left_and_right_mentions(train_set, is_gold=True)
        logging.info('Dev gold mentions - loading predicates and their arguments ')
        find_left_and_right_mentions(dev_set, is_gold=True)
        logging.info('Test gold mentions - loading predicates and their arguments ')
        find_left_and_right_mentions(test_set, is_gold=True)
        if config_dict["load_predicted_mentions"]:
            logging.info('Test predicted mentions - loading predicates and their arguments ')
            find_left_and_right_mentions(test_set, is_gold=False)

    # 9. load elmo
    if config_dict["load_elmo"]: # load ELMo embeddings
        elmo_embedder = ElmoEmbedding(config_dict["options_file"], config_dict["weight_file"])
        logging.info("Loading ELMO embeddings...")
        load_elmo_embeddings(train_set, elmo_embedder, set_pred_mentions=False)
        load_elmo_embeddings(dev_set, elmo_embedder, set_pred_mentions=False)
        load_elmo_embeddings(test_set, elmo_embedder, set_pred_mentions=True)

    # 10.
    logging.info('Storing processed data...')
    with open(os.path.join(args.output_path,'training_data'), 'wb') as f:
        cPickle.dump(train_set, f)
    with open(os.path.join(args.output_path,'dev_data'), 'wb') as f:
        cPickle.dump(dev_set, f)
    with open(os.path.join(args.output_path, 'test_data'), 'wb') as f:
        cPickle.dump(test_set, f)


if __name__ == '__main__':
    main(args)
