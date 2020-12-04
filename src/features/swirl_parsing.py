import os
from typing import Dict, List, Tuple, Union  # for type hinting
from src.shared.classes import Srl_info
# import sys
# for pack in os.listdir("src"):
#     sys.path.append(os.path.join("src", pack))
# sys.path.append("/src/shared/")
# from classes import *


def parse_swirl_sent(sent_id, sent_tokens):
    '''
    This function gets a sentence in a SwiRL "format" and extracts the predicates
    and their arguments from it.
    The function returns a dictionary in the following structure:
    dict[key3] = Srl_info object
    while key3 is a token id of an extracted event.
    See a documentation about Srl_info object in classed.py.
    :param sent_id: the sentence ordinal number in the document
    :param sent_tokens: the sentence's tokens
    :return: a dictionary as mentioned above
    '''
    col = 0
    event_dict = {}
    for tok_idx, tok in enumerate(sent_tokens):
        if tok[0] != '-':
            col += 1
            events_args = {}
            # look for the arguments
            for arg_idx, arg in enumerate(sent_tokens):
                if '(' in arg[col] and ')' in arg[col]:
                    # one word argument
                    arg_name = arg[col][1:-1]
                    arg_name = arg_name.replace('*','')
                    arg_name = arg_name.replace('R-', '')
                    events_args[arg_name] = [arg_idx]
                elif '(' in arg[col]:
                    # argument with two or more words
                    arg_bound_found = False
                    arg_name = arg[col][1:-1]
                    arg_name = arg_name.replace('*', '')
                    arg_name = arg_name.replace('R-', '')
                    events_args[arg_name] = [arg_idx]
                    bound_idx = arg_idx + 1
                    while bound_idx < len(sent_tokens) and not arg_bound_found:
                        if ')' in sent_tokens[bound_idx][col]:
                            events_args[arg_name].append(bound_idx)
                            arg_bound_found = True
                        bound_idx += 1
            # save the arguments per predicate
            event_dict[tok_idx] = Srl_info(sent_id, events_args,tok_idx, tok[0])
    return event_dict


def parse_swirl_file(xml_file_name, srl_file_path, srl_data):
    '''
    This function:
    - reads one SwiRL output files given by *srl_file_path*,
    - extracts the predicates and their arguments for each semantic role,
    - add those extracted srl info into *srl_data*.

    :param xml_file_name: name of the input file of SwiRL, (name of the xml file in ecb+)
    :param srl_file_path: path to the output file of SwiRL
    :param srl_data: A dict that saves srl info.
        This dict is used as return value.
        srl_data[doc_id][sent_id][predicted_token_id] = Srl_info object.
    '''
    doc_id = xml_file_name.split('.')[0]  # xml_file_name="1_1ecb.xml"  doc_id="1_1ecb"
    """xml_file_mainname equals to the doc_id of corresponding Document obj."""
    srl_data[doc_id] = {}

    sent_id = 0
    sent_tokenlist = []
    srl_file = open(srl_file_path, 'r')
    for token_line in srl_file:
        token_line = token_line.strip().split()
        if token_line:
            sent_tokenlist.append(token_line)
        else:
            srl_data[doc_id][sent_id] = parse_swirl_sent(sent_id, sent_tokenlist)
            sent_id += 1
            sent_tokenlist = []
    # parse the last sentence
    srl_data[doc_id][sent_id] = parse_swirl_sent(sent_id, sent_tokenlist)
    srl_file.close()


def parse_swirl_output(srl_folder_path: str) -> Dict[str, Dict[int, Dict[int, Srl_info]]]:
    '''
    This function reads all SwiRL output files into a return srl dict.
    This function:
    - reads one SwiRL all SwiRL output files given by *srl_folder_path*,
    - extracts the predicates and their arguments for each semantic role,
    - return the extracted srl info in a dict.

    The return dict likes: return[doc_id][sent_id] [predicted_token_id]=Srl_info obj.
    A predicted can include multi tokens, predicate_token_id is the id of the first
    token in the predicate.
    For example::

        return = {
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

    :param srl_folder_path: the path to the folder which includes the output files of SwiRL
    :return: a dictionary like: dict[doc_id][sent_id][token_id] = Srl_info object.
    '''
    srl_data = {}
    srl_file_name_list = os.listdir(srl_folder_path)
    for srl_file_name in srl_file_name_list:
        srl_file_path = os.path.join(srl_folder_path, srl_file_name)
        splitted = srl_file_name.split('.')  # 'SWIRL_OUTPUT.10_13ecbplus.xml.txt'
        xml_file_name = splitted[1] + '.' + splitted[2]  # '10_13ecbplus.xml'

        parse_swirl_file(xml_file_name, srl_file_path, srl_data)

    return srl_data


