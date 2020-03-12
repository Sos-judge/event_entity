# -*- coding: utf-8 -*-
# third part package
import os
import csv
import json
import _pickle as cPickle  # import _pickle as cPickle
import logging
import argparse
import xml.etree.ElementTree as ET  # for the parse of xml file.
# import Lib.traceback as traceback  # for try and catch
import sys
import os
from typing import Dict, List, Tuple  # for type hinting

# local package
from src.data.mention_data import MentionData

# 读取指令参数
parser = argparse.ArgumentParser(description='Parsing ECB+ corpus')
parser.add_argument('--ecb_path', type=str,
                    help=' The path to the ECB+ corpus')
parser.add_argument('--output_dir', type=str,
                        help=' The directory of the output files')
parser.add_argument('--data_setup', type=int,
                        help='Set the desirable dataset setup, 1 for Yang/Choubey setup and 2 for Cybulska/Kenyon-Dean setup (recommended)')
parser.add_argument('--selected_sentences_file', type=str,
                    help=' The path to a file contains selected sentences from the ECB+ corpus (relevant only for '
                         'the second evaluation setup (Cybulska setup)')
args = parser.parse_args()

# 建立输出路径
out_dir = args.output_dir
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# class Token(object):
#     '''
#     An helper class which represents a single when reading the corpus.
#     '''
#     def __init__(self, t_text, t_sentenceindex, t_tokenindex, rel_id=None):
#         '''
#
#         :param text: The token text
#         :param sent_id: The sentence id
#         :param tok_id: The token id
#         :param rel_id: The relation id (i.e. coreference chain)
#         '''
#
#         self.text = t_text
#         self.sent_id = t_sentenceindex
#         self.tok_id = t_tokenindex
#         self.rel_id = rel_id


def read_selected_sentences(filename: str) -> dict:
    """
    This function reads the CSV file that was released with ECB+ corpus and returns a
    dictionary contains those sentences IDs. This file contains the IDs of 1840 sentences
    which were manually reviewed and checked for correctness.
    The ECB+ creators recommend to use this subset of the dataset.

    :param filename: the CSV file
    :return: a dictionary, where a key is an XML filename (i.e. ECB+ document) and the value is a list contains all
        the sentences IDs that were selected from that XML filename.

    example1:
        filename = 'data\\\\raw\\\\ECBplus_coreference_sentences.csv'. and the content in this .csv file is like:
            1 10ecbplus 1  \n
            1 10ecbplus 3  \n
            1 11ecbplus 1  .
        return = {'1_10ecbplus.xml':[1,3],'1_11ecbplus.xml':[1]}
    """
    xml_to_sent_dict = {}
    with open(filename, 'rt') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        # reader.next()
        next(reader)
        for line in reader:
            xml_filename = '{}_{}.xml'.format(line[0], line[1])
            sent_id = int(line[2])

            if xml_filename not in xml_to_sent_dict:
                xml_to_sent_dict[xml_filename] = []
            xml_to_sent_dict[xml_filename].append(sent_id)

    return xml_to_sent_dict


def m_tag_to_type(tag: str) -> str:
    """
    example::
        'ACT' = type_to_type_abbr('ACTION_REPORTING')
        'UNK' = type_to_type_abbr('UNKNOWN_INSTANCE_TAG')

    :param tag:  m_tag.
    :return: m_type.
    """
    if tag == 'UNKNOWN_INSTANCE_TAG':
        return 'UNK'
    if 'ACTION' in tag:
        return 'ACT'
    if 'LOC' in tag:
        return 'LOC'
    if 'NON' in tag:
        return 'NON'
    if 'HUMAN' in tag:
        return 'HUM'
    if 'TIME' in tag:
        return 'TIM'
    else:
        print('unknown tag:', tag)


def i_id_to_i_type(i_id: str) -> str:
    '''
    example::
        i_id_to_i_type('ACT150798320') -> 'ACT'

    :param i_id: i_id or anything that like a i_id( e.g. note property in cd r， instance_id
        property in cd tm).
    :return: type
    '''
    if 'ACT' in i_id or 'NEG' in i_id:
        return 'ACT'
    if 'LOC' in i_id:
        return 'LOC'
    if 'NON' in i_id:
        return 'NON'
    if 'HUM' in i_id or 'CON' in i_id:
        return 'HUM'
    if 'TIM' in i_id:
        return 'TIM'
    if 'UNK' in i_id:
        return 'UNK'


def sentencindex_to_bodysentenceindex(doc_id: str, sentenceindex: int, parse_all: bool) -> int:
    '''
    in xml file, a news text = [newsUrl] + [newsTime] + newsBody

    In X_Xecb.xml, news body starts with sentence 0. So sm_bodysentenceindex = sm_sentenceindex.
    A sm_bodysentenceindex is different from sm_sentenceindex in 2 condition:
        - 1st: Only in X_Xecbplus.xml, there is a newsUrl which is the sentence 0. So, newsBody
          starts with sentence 1.
        - 2nd: Only in 9_3ecbplus.xml and 9_4ecbplus.xml, there is a newsTime which is the sentence
          1. So, newsBody starts with sentence 2.
    '''
    bodysentenceindex = sentenceindex
    # 1st condition: there is url sentence.
    if 'plus' in doc_id:
        if int(bodysentenceindex) > 0:
            bodysentenceindex -= 1
    # 2nd condition: there is time sentence.
    if parse_all and (doc_id == '9_3ecbplus' or doc_id == '9_4ecbplus'):
        if bodysentenceindex > 0:
            bodysentenceindex -= 1
    return bodysentenceindex


def calc_split_statistics(dataset_split, split_name, statistics_file_name):
    '''
    This function calculates and saves the statistics of a split (train/dev/test) into a file.
    :param dataset_split: a list that contains all the mention objects in the split
    :param split_name: the split name (a string)
    :param statistics_file_name: a filename for the statistics file
    '''
    event_mentions_count = 0
    human_mentions_count = 0
    non_human_mentions_count = 0
    loc_mentions_count = 0
    time_mentions_count = 0
    non_continuous_mentions_count = 0
    unk_coref_mentions_count = 0
    coref_chains_dict = {}

    for mention_obj in dataset_split:
        mention_type = mention_obj.mention_type
        if 'ACT' in mention_type or 'NEG' in mention_type:
            event_mentions_count += 1
        elif 'NON' in mention_type:
            non_human_mentions_count += 1
        elif 'HUM' in mention_type:
            human_mentions_count += 1
        elif 'LOC' in mention_type:
            loc_mentions_count += 1
        elif 'TIM' in mention_type:
            time_mentions_count += 1
        else:
            print(mention_type)

        is_continuous = mention_obj.is_continuous
        if not is_continuous:
            non_continuous_mentions_count += 1

        coref_chain = mention_obj.coref_chain
        if 'UNK' in coref_chain:
            unk_coref_mentions_count += 1
        if coref_chain not in coref_chains_dict:
            coref_chains_dict[coref_chain] = 1
    with open(statistics_file_name, 'a') as f:

        f.write('{} statistics\n'.format(split_name))
        f.write('-------------------------\n')
        f.write( 'Number of event mentions - {}\n'.format(event_mentions_count))
        f.write( 'Number of human participants mentions - {}\n'.format(human_mentions_count))
        f.write( 'Number of non-human participants mentions - {}\n'.format(non_human_mentions_count))
        f.write( 'Number of location mentions - {}\n'.format(loc_mentions_count))
        f.write( 'Number of time mentions - {}\n'.format(time_mentions_count))
        f.write( 'Total number of mentions - {}\n'.format(len(dataset_split)))

        f.write( 'Number of non-continuous mentions - {}\n'.format(non_continuous_mentions_count))
        f.write( 'Number of mentions with coref id = UNK - {}\n'.format(unk_coref_mentions_count))
        f.write( 'Number of coref chains = {}\n'.format(len(coref_chains_dict)))
        f.write('\n')


def save_gold_mention_statistics(train_extracted_mentions, dev_extracted_mentions,
                                  test_extracted_mentions):
    '''
    This function calculates and saves the statistics of each split (train/dev/test) into a file.
    :param train_extracted_mentions: a list that contains all the mention objects in the train split
    :param dev_extracted_mentions: a list that contains all the mention objects in the dev split
    :param test_extracted_mentions: a list that contains all the mention objects in the test split
    '''
    logger.info('Calculate mention statistics...')

    all_data_mentions = train_extracted_mentions + dev_extracted_mentions +test_extracted_mentions
    filename = 'mention_stats.txt'
    calc_split_statistics(train_extracted_mentions, 'Train set',
                          os.path.join(args.output_dir,filename))

    calc_split_statistics(dev_extracted_mentions, 'Dev set',
                            os.path.join(args.output_dir, filename))

    calc_split_statistics(test_extracted_mentions, 'Test set',
                          os.path.join(args.output_dir, filename))

    calc_split_statistics(all_data_mentions, 'Total',
                          os.path.join(args.output_dir, filename))

    logger.info('Save mention statistics...')


def read_ecb_plus_doc(selected_sent_list: List[int],
                      doc_name: str, doc_id: str,
                      output_file_obj,
                      extracted_mentions: List,
                      parse_all: bool, load_singletons: bool):

    """
    Read xml file of ecb+ corpora, extract t info and sm info, and save it.

    ---------------
     t info
    ---------------
    t info are saved into the text file given by *output_file_obj*.

    - A t will be saved when the following statement is True::

        parse_all or (t_sentenceindex in selected_sent_list)

    - In the file, one line corresponds to one t.
      There is one black line after the end token of a sentence.
      And there are 5 columns:

      - the 1st column: *doc_id* parameter of this function.
      - the 2nd column: t_bodysentenceindex. For detail , see sentencindex_to_bodysentenceindex
      - the 3rd column: t_id.
      - the 4th column: t_text, and if t_text is ' ' or '\t', it will be replaced by a '-'.
      - The 5th column: the id of the i that this t refer to.

        - '-', if this t isn't in a mention( so it do not refer to any i).
        - wd_i_id( For detail, see wd_i_info.), if this t refer to a wd i.
        - cd_i_id( For detail, see cd_i_info.), if this t refer to a cd i.
        - sg_i_id( For detail, see sg_i_info.), if this t refer to a sg i and load singletons;
        - '-', if this t refer to a sg i and do not load singletons;

      example::

            14_5ecb	2	37	being	-
            14_5ecb	2	38	treated	ACT17478359686333015
            14_5ecb	2	39	as	-
            14_5ecb	2	40	potentially	-
            14_5ecb	2	41	suspicious	-
            14_5ecb	2	42	.	-
            14_5ecb	2	43	''	-

            14_6ecb	0	0	Residents	Singleton_HUM_14_14_6ecb
            14_6ecb	0	1	evacuated	Singleton_ACT_15_14_6ecb
            14_6ecb	0	2	from	-
            14_6ecb	0	3	their	INTRA_UNK_33690_14_6ecb
            14_6ecb	0	4	homes	INTRA_UNK_33690_14_6ecb
            14_6ecb	0	5	after	-
            14_6ecb	0	6	a	-
            14_6ecb	0	7	huge	-
            14_6ecb	0	8	fire	ACT17478306085573007

    ---------------
     sm info
    ---------------
    sm info: are saved into parameter *extracted_mentions* which is a shallow copied object.

    - A sm will be saved when the following statement is True::

        parse_all or (sm_sentenceindex in selected_sent_list)

    - sm info is saved as a *src.mention_data.MentionData* objects, which has the following property：

      - doc_id = doc_id
      - cur_t_sentenceindex = cur_sm_bodysentenceindex
      - tokens_numbers = cur_sm_tokenindex
      - tokens_str = cur_sm_string
      - coref_chain = cur_i_id
      - mention_type = cur_sm_type,
      - is_continuous = is_continuous
      - is_singleton = is_singleton
      - score = float(-1)

    :param selected_sent_list: A list of the sentenceindex( rather than a bodysentenceindex)
        of 'selected sentence'.
        This parameter is activated only when parse_all= False, and then extract
        info from only those 'selected sentences'.
    :param doc_name: path of the xml file, e.g. 'data\\raw\\ECBplus\\1\\1_10ecb.xml'
        The xml file should be encoded with utf-8.
    :param doc_id: document ID of the xml file, in form of  {topic id}_{file name}{ecb/ecbplus type}).
        e.g. '1_10ecb'.
    :param output_file_obj: In the 1st way, info is saved into a text file, and this parameter
        is the io stream object(the return of open() function) of this text file.
    :param extracted_mentions: Info is saved into this list in the 2th way. This list
        accumulates the extracted info from multiple calls to this function on every
        xml file.
        In principle, any value is ok for this parameter, because this function only
        add info to it. But usually the value is the info that extracted in the previous
        calling of this function.
        The info added is in the form of src.data.mention_data.MentionData object
    :param parse_all: From which sentence to extract information
        True, extract info from all sentences in xml file as in Yang setup;
        False, extract info from only the selected sentences as in Cybulska setup.
            The selected sentences is given by parameter selected_sent_list.
    :param load_singletons:  a boolean variable indicates whether to read singleton mentions as in
        Cybulska setup or whether to ignore them as in Yang setup.
    """
    ecb_file = open(doc_name, 'r', encoding='UTF-8')
    tree = ET.parse(ecb_file)
    root = tree.getroot()

    t_info: dict[str, dict] = {}
    """
    {tid: t_info_dict}
        - tid: 't_id' attr of a token, all tokens(selected or not) are included.
        - t_info_dict['t_text']: text of this token, e.g. 'has'
        - t_info_dict['t_sentenceindex']:  'sentence' attr of this token, it is the index of 
          this sentence start with 1st sentence in whole doc. e.g. "3"
        - t_info_dict['t_bodysentenceindex']:  'sentence' attr of this token, it is the index of
          this sentence start with 1st sentence in news body. e.g. "1"
        - t_info_dict['t_tokenindex']: 'number' attr of this token, e.g. "0"
        - t_info_dict['i_id']: id of the i that this t refer to.
          - wd_i_id( For detail, see wd_i_info.) for a wd i. 
          - cd_i_id( For detail, see cd_i_info.) for a cd i. 
          - sg_i_id( For detail, see sg_i_info.) for a sg i, if load singletons; 
            None, if not load singletons.
        - i_desc: description of the i that this t refer to.
          - wd_i_desc for a wd i. For detail, see wd_i_info.
          - cd_i_desc for a cd i. For detail, see cd_i_info.
          - 'padding' for a sg i, if load singletons; None, if not load singleton.
    """

    sm_info: dict[str, str] = {}
    """
    {sm_id: sm_tag}
        - sm_id: 'm_id' attr of a sm. All(selected and not; sg, wd and cd) sm are included.
        - sm_tag: tag of this sm.
    """

    tm_info: dict[str, tuple[str, str]] = {}
    """
    {'tm_id':(tm_desc, tm_iid_or_tag)}
        - tm_id: 'm_id' attr of a tm. All( selected and not) cd and wd tm are included.
        - tm_desc: 'TAG_DESCRIPTOR' attr of the tm.          
        - tm_iid_or_tag: 
        
          - for CD tm, it is the 'instance_id' attr of this tm.
          - for WD tm, it is the tag of this tm

    """

    mapped_sm_id: [str] = []
    """
    [sm_id]
        - sm_id: 'm_id' attr of sm which refer to a tm(seleted and not; CD and WD tm,
          except SG tm).
    """

    sm_id_to_t_id: dict[str, list[str]] = {}
    """
    {sm_id:[t_id,t_id,...]}
        - sm_id: 'm_id' attr of a sm，all sm included( selected and not; sg sm, wd sm and
          cd sm).
        - t_id: 't_id' attr of token of this sm.
    """

    sm_id_to_i_id: dict[str, str] = {}
    """
    {sm_id: i_id}
        - sm_id: 'm_id' attr of a sm, all sm included( selected or not; cd sm, wd sm and 
          sg sm).
        - i_id: 
        
          - a wd_i_id(for detail, see wd_i_info) for a wd i. 
          - a cd_i_id(for detail, see cd_i_info) for a cd i.
          - a sg_i_if(for detail, see sg_i_info) for a sg i, if load_singletons=True.
    """

    wd_i_info: dict[str, tuple[list[str], str]] = {}
    """
    {wd_i_id:(wd_i_tokenlist, wd_i_desc)}
        - wd_i_id: represented in the form of 'INTRA_{wd_i_type}_rId_docId'.
          
          - wd_i_type: A wd_i has many tm and sm, each of them has a type. wd_i_type
            equals to the type if all the types are same; otherwise, a accordance strategy
            is need:
            
            - tm_type strategy: set wd_i_type as wd_tm_type without accordance check
            - Barhom2019 strategy: 
              this value is 'UNK' if the type of tm is 'UNKNOWN_INSTANCE_TAG', 
              otherwise, the value equals to the type of the first sm.
              
          - rId: A wd_i has only one wd_r, rId is the 'r_id' attr of this wd_r
          - docId:
          
        - wd_i_tokenlist:[t_id, t_id, ...], t_id of all(in cur doc) the t that refer to this wd_i.
        - wd_i_desc: description string of this wd_i.
          The value equals to 'TAG_DESCRIPTOR' attr of the tm of this wd_i.
    """

    cd_i_info: dict[str, tuple[list[str], str]] = {}
    """
    {'cd_i_id': (cd_i_tokenlist, cd_i_desc)}
        - cd_i_id: if the 'instance_id ' attr of corresponding tm and the 'note'
          attr of the corresponding r is equal, cd_i_id is this value. otherwise, a 
          accordance strategy is need:
          
        - cd_i_tokenlist:[t_id, t_id, ...], t_id of all(in cur doc) the t that refer to
          this cd_i.
        - cd_i_desc: description string of this cd_i.
          The value equals to 'TAG_DESCRIPTOR' attr of the tm of this cd_i.
    """

    sg_i_info: dict[str, tuple[list[str], str]] = {}
    """
    there is sg_i only when load_singletons=True
    {'sg_i_id': (g_i_tokenlist, sg_i_desc)}
        - sg_i_id: in the form of 'Singleton_{sg_i_type}_{sg_sm_id}_{doc_id}'
        
          - sg_i_type: equals to the type of sg_sm.( A sg_i has only one sg_sm, no 
            accordance check needed)
          - sg_sm_id: id of the sg_sm.( A sg_i has only one sg_sm, no accordance check 
            needed)
          - doc_id:

        - sg_i_tokenlist:[t_id, t_id, ...], t_id of all(in cur doc) the t that refer to
          this sg_i. But it is None because we do not need this, the varable sg_i_tokenlist
          here is to keep the same structer as cd_i_info and wd_i_info.
        - sg_i_desc: description string of this cd_i.But it is None because there is no
          description for a sg i in ecb+, the varable sg_i_desc here is to keep the same
          structer as cd_i_info and wd_i_info.
    """

    # 1.extract info from <Markables>...</Markables> tag
    """
    iterate through every tag in <Markables>...</Markables>
    there are 4 kinds of conditions and 3 kinds of mention marked as (1),(2)source
    mention,(3)CD target mention,(4)WD target mention
        <Markables>
        (2) <XXX m_id="48" note="byCROMER" >
        (1)     <token_anchor t_id="19"/>
        (1)     <token_anchor t_id="20"/>
            </XXX>
        (3) <TIME_DURATION m_id="52" RELATED_TO="" TAG_DESCRIPTOR="t26_decades" instance_id="TIM18440826675897964" />
        (3) <ACTION_OCCURRENCE m_id="51" RELATED_TO="" TAG_DESCRIPTOR="t26_died" instance_id="ACT18440577880137709" />
        (4) <UNKNOWN_INSTANCE_TAG m_id="17" RELATED_TO="" TAG_DESCRIPTOR="" />"
        (4) <HUMAN_PART_PER m_id="40" RELATED_TO="" TAG_DESCRIPTOR="spokesman" />
        </Markables>
    """
    cur_m_id = ''
    for action in root.find('Markables').iter():
        if action.tag == 'Markables':
            continue
        # for condition (1)
        elif action.tag == 'token_anchor':
            sm_id_to_t_id[cur_m_id].append(action.attrib['t_id'])
        # for condition (2)(3)(4)
        else:
            cur_m_id = action.attrib['m_id']
            # for condition (3)(4), it is a tm
            if 'TAG_DESCRIPTOR' in action.attrib:
                # for condition (3), it is a CD tm
                if 'instance_id' in action.attrib:
                    tm_info[cur_m_id] = (
                        action.attrib['TAG_DESCRIPTOR'], action.attrib['instance_id'])
                # for condition (4), it is a WD tm
                else:
                    tm_info[cur_m_id] = (
                        action.attrib['TAG_DESCRIPTOR'], action.tag)
            # for condition (2), it is a source mention
            else:
                sm_id_to_t_id[cur_m_id] = []
                sm_info[cur_m_id] = action.tag

    # 2.extract info from <Relations><INTRA_DOC_COREF>...</INTRA_DOC_COREF></Relations>
    """
    iterate through every tag in <Markables><INTRA_DOC_COREF>...</INTRA_DOC_COREF></Markables>
    there are 3 kinds of conditions marked as (i1)(i2)(i3)
        <Relations>
        (i1) <INTRA_DOC_COREF r_id="37615" >
        (i2)    <source m_id="24" />
        (i2)    <source m_id="25" />
        (i2)    <target m_id="17" />
             </INTRA_DOC_COREF>
        </Relations>
    """
    cur_wd_i_id = ''
    cur_wd_i_tokenlist= []
    for cur_wd_r in root.find('Relations').findall('INTRA_DOC_COREF'):
        for child in cur_wd_r.iter():
            # for condition (i1), cur r is a WD r
            if child.tag == 'INTRA_DOC_COREF':
                # two strategy for setting the cur_wd_i_type, leave the accordance check for later.
                strategy = 'Barhom2019 strategy'
                if strategy == 'Barhom2019 strategy':
                    cur_wd_tm_tag = tm_info[cur_wd_r.find('target').get('m_id')][1]
                    # this value is 'UNK' if the type of tm is 'UNKNOWN_INSTANCE_TAG'
                    if cur_wd_tm_tag == 'UNKNOWN_INSTANCE_TAG':
                        cur_wd_i_type = m_tag_to_type(cur_wd_tm_tag)
                    # otherwise, the value equals to the type of the first sm.
                    else:
                        cur_1th_wd_sm_tag = sm_info[cur_wd_r.find('source').get('m_id')]
                        cur_1th_wd_sm_type = m_tag_to_type(cur_1th_wd_sm_tag)
                        cur_wd_i_type = cur_1th_wd_sm_type
                    # cid of a WD coref is represented by 'INTRA_tagAttr_rId_docId'
                elif strategy == 'tm_type strategy':
                    # tm_type strategy:
                    # set wd_i_type as wd_tm_type immediately
                    cur_wd_tm_tag = tm_info[cur_wd_r.find('target').get('m_id')][1]
                    cur_wd_tm_type = m_tag_to_type(cur_wd_tm_tag)
                    cur_wd_i_type = cur_wd_tm_type
                cur_wd_i_id = 'INTRA_{}_{}_{}'.format(cur_wd_i_type, child.attrib['r_id'], doc_id)
                wd_i_info[cur_wd_i_id] = ()
            # for condition (i2), it is the sm in cur r
            elif child.tag == 'source':
                cur_wd_i_tokenlist += (sm_id_to_t_id[child.attrib['m_id']])
                mapped_sm_id.append(child.attrib['m_id'])
                sm_id_to_i_id[child.attrib['m_id']] = cur_wd_i_id
            # for condition (i3), it is the tm in cur r
            else:
                wd_i_info[cur_wd_i_id] = (cur_wd_i_tokenlist, tm_info[child.attrib['m_id']][0])
                # end of iteration of cur relation, clear variable for iteration of the next relation.
                cur_wd_i_tokenlist = []

    # 3. extract info from <Relations><CROSS_DOC_COREF>...</CROSS_DOC_COREF></Relations>
    """
    iterate through every tag in <Markables><CROSS_DOC_COREF>...</CROSS_DOC_COREF></Markables>
    there are 3 kinds of conditions marked as (c1)(c2)(c3)
        <Relations>
        (c1) <CROSS_DOC_COREF r_id="37623" note="ACT16235311629112331" >
        (c2)      <source m_id="36" />
        (c2)      <source m_id="37" />
        (c3)      <target m_id="49" />
              </CROSS_DOC_COREF>
        </Relations>
    """
    cur_cd_i_id = ''
    cur_cd_i_tokenlist = []
    for cross_doc_coref in root.find('Relations').findall('CROSS_DOC_COREF'):
        for child in cross_doc_coref.iter():
            # for condition (c1), cur r is CD r
            if child.tag == 'CROSS_DOC_COREF':
                # set the cd_i_id as r_note immediately, leave the accordance check for later.
                cur_cd_r_note = child.attrib['note']
                cur_cd_i_id = cur_cd_r_note
                cd_i_info[cur_cd_i_id] = ()
            # for condition (c2), it is the sm in cur r
            elif child.tag == 'source':
                cur_cd_i_tokenlist += (sm_id_to_t_id[child.attrib['m_id']])
                mapped_sm_id.append(child.attrib['m_id'])
                sm_id_to_i_id[child.attrib['m_id']] = cur_cd_i_id
            # for condition (c3), it is the tm in cur r
            else:
                cd_i_info[cur_cd_i_id] = (
                    cur_cd_i_tokenlist, tm_info[child.attrib['m_id']][0])
                # end of iteration of cur relation, clear variable for iteration of the next relation.
                cur_cd_i_tokenlist = []

    # 4. extract info from <token>...</token>
    for cur_t in root.findall('token'):
        t_info[cur_t.attrib['t_id']] = {
            't_text': cur_t.text,
            't_sentenceindex': cur_t.attrib['sentence'],
            't_tokenindex': cur_t.attrib['number'],
            'i_id': None,
            'i_desc': None
        }
    for cur_cd_i_id in cd_i_info:
        for cur_t_id in cd_i_info[cur_cd_i_id][0]:
            t_info[cur_t_id]['i_id'] = cur_cd_i_id
            t_info[cur_t_id]['i_desc'] = cd_i_info[cur_cd_i_id][1]
    for cur_wd_i_id in wd_i_info:
        for cur_t_id in wd_i_info[cur_wd_i_id][0]:
            t_info[cur_t_id]['i_id'] = cur_wd_i_id
            t_info[cur_t_id]['i_desc'] = wd_i_info[cur_wd_i_id][1]

    # 5. Load singletons if required
    if load_singletons:
        # find the sg sm(singleton source mention is sm that isn't refer to any cd tm or wd tm)
        for mid in sm_id_to_t_id:
            if mid not in mapped_sm_id:
                # sm in sm_id_to_t_id include sg, cd and wd sm.
                # sm in mapped_sm_id include cd and wd sm.
                # so, the sg sm are the difference between them.
                # iterate through every sg sm
                cur_sg_sm_id = mid

                # 1. create instance id for each singleton mention
                cur_sg_sm_tag = sm_info[cur_sg_sm_id]
                cur_sg_sm_type = m_tag_to_type(cur_sg_sm_tag)
                cur_sg_i_type = cur_sg_sm_type  # a sg_i has only one sg_sm, no accordance check needed.
                cur_sg_i_id = 'Singleton_{}_{}_{}'.format(cur_sg_i_type, cur_sg_sm_id, doc_id)
                sg_i_info[cur_sg_i_id] = ()
                # 2. updated sm_id_to_i_id
                # this mention is related to the singleton instance, so this mention appears
                # in this singleton coref, so this mention can be listed in sm_id_to_i_id
                sm_id_to_i_id[cur_sg_sm_id] = cur_sg_i_id
                # 3. updated tokens
                # the token of this mention had it rel_id property as None, after this mention
                # is related to the singleton instance, the rel_id property should save the info
                # of the singleton instance.
                unmapped_tids = sm_id_to_t_id[cur_sg_sm_id]
                for cur_t_id in unmapped_tids:
                    if t_info[cur_t_id]['i_id'] is None:
                        t_info[cur_t_id]['i_id'] = cur_sg_i_id
                        t_info[cur_t_id]['i_desc'] = 'padding'

    # 6. sm info is saved into *extracted_mentions*
    for cur_sm_id in sm_id_to_t_id:
        cur_sm_sentenceindex = int(t_info[
                                       sm_id_to_t_id[cur_sm_id][0]
                                   ]['t_sentenceindex'])
        '''
        check if cur mention need to be save.
        if user want to parse all sentences, then process cur mention
        if user want to parse only the selected sentences, and cur mention is
            selected，then process cur mention
        '''
        if not (parse_all or (cur_sm_sentenceindex in selected_sent_list)):
            continue

        # (1
        cur_sm_tag = sm_info[cur_sm_id]
        cur_sm_type = m_tag_to_type(cur_sm_tag)

        # (2
        cur_i_id = sm_id_to_i_id[cur_sm_id]
        cur_i_type = i_id_to_i_type(cur_i_id)

        # ---------------------------------------------------------------------
        # the 2 types above should be same
        # if cur_sm_type != cur_i_type:
        #     print('err: diff types in same coref: {}'.format(cur_i_id))
        #     print('  type_attr_of_cur_c: {}'.format(cur_i_type))
        #     print('  type_attr_of_cur_m: {}'.format(cur_sm_type))
        # ------------------------------------------------------------------------

        # (3
        cur_sm_tokenindex = []  # One sm has some t, t has *number* attr( token index), this is a list of every *number*
        cur_sm_textlist = []
        tids = sm_id_to_t_id[cur_sm_id]
        for cur_t_id in tids:
            cur_t = t_info[cur_t_id]
            if int(cur_t['t_tokenindex']) not in cur_sm_tokenindex:
                cur_sm_tokenindex.append(int(cur_t['t_tokenindex']))
                cur_sm_textlist.append(cur_t['t_text'])  # .encode('ascii', 'ignore')修改了这里
        cur_sm_string = ' '.join(cur_sm_textlist)

        # (4
        cur_sm_bodysentenceindex = sentencindex_to_bodysentenceindex(
            doc_id, cur_sm_sentenceindex, parse_all
        )

        # (5
        is_continuous = True if cur_sm_tokenindex == range(cur_sm_tokenindex[0], cur_sm_tokenindex[-1]+1) else False

        # (6
        is_singleton = True if 'Singleton' in cur_i_id else False

        # create mention obj based on the above info, and add it to extracted_mentions
        mention_obj = MentionData(doc_id,
                                  cur_sm_bodysentenceindex,
                                  cur_sm_tokenindex,
                                  cur_sm_string,
                                  cur_i_id,
                                  cur_sm_type,
                                  is_continuous=is_continuous,
                                  is_singleton=is_singleton,
                                  score=float(-1))
        extracted_mentions.append(mention_obj)

    # 7. t info is saved into text file
    prev_t_bodysentenceindex = None  # bodysentenceindex of previous sentence

    for cur_t_id in t_info:
        cur_t = t_info[cur_t_id]
        cur_t_sentenceindex = int(cur_t['t_sentenceindex'])

        if not parse_all and cur_t_sentenceindex not in selected_sent_list:
            continue

        cur_t_tokenindex = int(cur_t['t_tokenindex'])
        cur_t_text = cur_t['t_text'] if (cur_t['t_text'] != '' and cur_t['t_text'] != '\t') else '-'
        cur_t_iid = cur_t['i_id']
        cur_t_bodysentenceindex = sentencindex_to_bodysentenceindex(
            doc_id, cur_t_sentenceindex, parse_all
        )

        # write into output file: if go to next sentence, go to next line
        if prev_t_bodysentenceindex is None or prev_t_bodysentenceindex != cur_t_bodysentenceindex:
            output_file_obj.write('\n')
            prev_t_bodysentenceindex = cur_t_bodysentenceindex
        # write into output file: token info
        s = doc_id \
            + '\t' + str(cur_t_bodysentenceindex) \
            + '\t' + str(cur_t_tokenindex) \
            + '\t' + cur_t_text \
            + '\t' + (cur_t_iid if cur_t_iid is not None else '-') \
            + '\n'
        output_file_obj.write(s)


def obj_dict(obj):
    return obj.__dict__


def save_split_mentions_to_json(split_name, mentions_list):
    '''
    This function gets  a mentions list of a specific split and saves its mentions in a JSON files.
    Note that event and entity mentions are saved in separate files.
    :param split_name: the split name
    :param mentions_list: the split's extracted mentions list
    '''
    event_mentions = []
    entity_mentions = []

    for mention_obj in mentions_list:
        mention_type = mention_obj.mention_type
        if 'ACT' in mention_type or 'NEG' in mention_type:
            event_mentions.append(mention_obj)
        else:
            entity_mentions.append(mention_obj)

    json_event_filename = os.path.join(args.output_dir, 'ECB_{}_Event_gold_mentions.json'.format(split_name))
    json_entity_filename =  os.path.join(args.output_dir, 'ECB_{}_Entity_gold_mentions.json'.format(split_name))

    with open(json_event_filename, 'w') as f:
        json.dump(event_mentions, f, default=obj_dict, indent=4, sort_keys=True)

    with open(json_entity_filename, 'w') as f:
        json.dump(entity_mentions, f, default=obj_dict, indent=4, sort_keys=True)


def parse_selected_sentences(xml_to_sent_dict: dict, parse_all: bool,
                             load_singletons: bool, data_setup: int):
    """
    xxx

    :param xml_to_sent_dict: selected sentences dictionary. This parameter is effective if parse_all=False.
    :param parse_all: a boolean variable indicates whether to read all the ECB+ corpus as in
        Yang setup or whether to filter the sentences according to a selected sentences list
        as in Cybulska setup.
    :param load_singletons:  boolean variable indicates whether to read singleton mentions as in
        Cybulska setup or whether to ignore them as in Yang setup.
    :param data_setup: the variable indicates the evaluation setup (which topics is for dev set
        and which is for train set) - 1 for Yang and Choubey setup and 2 for Cybulska Kenyon-Dean setup (recommended).
    """
    if data_setup == 1:  # Yang setup
        train_topics = range(1, 23)
        dev_topics = range(23, 26)
    else:  # Cybulska setup
        dev_topics = [2, 5, 12, 18, 21, 23, 34, 35]
        train_topics = [i for i in range(1, 36) if i not in dev_topics]  # train topics 1-35 , test topics 36-45

    dirs = os.listdir(args.ecb_path)  # ['1', '10', '11', ...]
    dirs_int = [int(d) for d in dirs]  # [1, 10, 11, ...]

    # empty list
    # classify all the ecb/ecb+ docs into train/dev/test set in sorted order.
    train_ecb_files_sorted = []  # [([0, 3], 'data\\raw\\ECBplus\\1\\1_10ecb.xml', '1_10ecb'),...]
    dev_ecb_files_sorted = []
    test_ecb_files_sorted = []
    train_ecb_plus_files_sorted = []
    dev_ecb_plus_files_sorted = []
    test_ecb_plus_files_sorted = []
    # traverse the topics, and fill the above list
    for topic in sorted(dirs_int):
        # cur topic
        cur_topic_id = str(topic)
        # get paths of docs in cur topic
        doc_files = os.listdir(os.path.join(args.ecb_path, cur_topic_id))
        # classify the docs: ecb， ecb+
        ecb_files = []
        ecb_plus_files = []
        for doc_file in doc_files:
            if 'plus' in doc_file:
                ecb_plus_files.append(doc_file)
            else:
                ecb_files.append(doc_file)
        # sort the docs
        ecb_files = sorted(ecb_files)
        ecb_plus_files = sorted(ecb_plus_files)
        # traverse the ecb docs, add info to train/test/dev_ecb_files_sorted list
        for ecb_file in ecb_files:
            # if user want to parse all sentences, then process cur doc
            # if user want to parse only the selected sentences, and cur doc includes at
            #   least 1 selected sentence, then process cur doc
            if parse_all or (ecb_file in xml_to_sent_dict):
                # get the relative path of xml file of cur doc
                xml_file_path = os.path.join(os.path.join(args.ecb_path, cur_topic_id), ecb_file)
                # get the selected sentence id in cur doc
                if parse_all:
                    selected_sentences = None
                else:
                    selected_sentences = xml_to_sent_dict[ecb_file]
                # classify cur doc into train/dev/test
                if topic in train_topics:
                    train_ecb_files_sorted.append((selected_sentences, xml_file_path,
                                                   ecb_file.replace('.xml', '')))
                elif topic in dev_topics:
                    dev_ecb_files_sorted.append((selected_sentences, xml_file_path,
                                                   ecb_file.replace('.xml', '')))
                else:
                    test_ecb_files_sorted.append((selected_sentences, xml_file_path,
                                                  ecb_file.replace('.xml', '')))
        # traverse the ecb+ docs, add info to train/test/dev_ecb_plus_files_sorted list
        for ecb_file in ecb_plus_files:
            if parse_all or ecb_file in xml_to_sent_dict:
                xml_file_path = os.path.join(os.path.join(args.ecb_path,cur_topic_id),ecb_file)
                if parse_all:
                    selected_sentences = None
                else:
                    selected_sentences = xml_to_sent_dict[ecb_file]
                if topic in train_topics:
                    train_ecb_plus_files_sorted.append((selected_sentences,
                                                        xml_file_path, ecb_file.replace('.xml', '')))
                elif topic in dev_topics:
                    dev_ecb_plus_files_sorted.append((selected_sentences,
                                                        xml_file_path, ecb_file.replace('.xml', '')))
                else:
                    test_ecb_plus_files_sorted.append(
                        (selected_sentences, xml_file_path, ecb_file.replace('.xml', '')))
    # combine the above list
    train_files = train_ecb_files_sorted + train_ecb_plus_files_sorted
    test_files = test_ecb_files_sorted + test_ecb_plus_files_sorted
    dev_files = dev_ecb_files_sorted + dev_ecb_plus_files_sorted

    # open file ECB_Dev/Train/Test_corpus.txt
    dev_out = open(os.path.join(args.output_dir, 'ECB_Dev_corpus.txt'), 'w', encoding='UTF-8')
    train_out = open(os.path.join(args.output_dir, 'ECB_Train_corpus.txt'), 'w', encoding='UTF-8')
    test_out = open(os.path.join(args.output_dir, 'ECB_Test_corpus.txt'), 'w', encoding='UTF-8')
    # empty list
    train_extracted_mentions = []
    dev_extracted_mentions = []
    test_extracted_mentions = []
    # fill the above list, and write the above file.
    for doc in train_files:
        read_ecb_plus_doc(doc[0], doc[1], doc[2], train_out, train_extracted_mentions, parse_all, load_singletons)
    for doc in dev_files:
        read_ecb_plus_doc(doc[0], doc[1], doc[2], dev_out, dev_extracted_mentions, parse_all, load_singletons)
    for doc in test_files:
        read_ecb_plus_doc(doc[0], doc[1], doc[2], test_out, test_extracted_mentions, parse_all, load_singletons)
    # close the above file
    train_out.close()
    dev_out.close()
    test_out.close()

    save_gold_mention_statistics(train_extracted_mentions, dev_extracted_mentions,
                                 test_extracted_mentions)

    save_split_mentions_to_json('Train', train_extracted_mentions)
    save_split_mentions_to_json('Dev', dev_extracted_mentions)
    save_split_mentions_to_json('Test', test_extracted_mentions)

    all_mentions = train_extracted_mentions + dev_extracted_mentions + test_extracted_mentions
    save_split_mentions_to_json('All', all_mentions)


def main():
    """
        This script processes the ECB+ XML files and saves for each data split (train/dev/test):
        1) A json file contains its mention objects.
        2) text file contains its sentences.
        .
        Runs data processing scripts to turn raw data from (../raw) into
        intermediate data (mention objects and sentences' text) ready for feature extraction
        (saved in ../intermid).
    """
    logger.info('Read ECB+ files')
    if args.data_setup == 1:  # Reads the full ECB+ corpus without singletons (Yang setup)
        parse_selected_sentences(xml_to_sent_dict={}, parse_all=True, load_singletons=False, data_setup=1)
    elif args.data_setup == 2:  # Reads the a reviewed subset of the ECB+ (Cybulska setup)
        xml_to_sent_dict = read_selected_sentences(args.selected_sentences_file)
        parse_selected_sentences(xml_to_sent_dict=xml_to_sent_dict,
                                 parse_all=False,
                                 load_singletons=True,
                                 data_setup=2)
    logger.info('ECB+ Reading was done.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    main()
