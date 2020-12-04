import os
import sys
import spacy
from typing import Dict, List, Tuple, Union  # for type hinting
# sys.path.append("/src/shared/")
# for pack in os.listdir("src"):
#     sys.path.append(os.path.join("src", pack))
from src.shared.classes import Corpus, Topic, Document, Sentence, Token
from spacy.tokens import Token as spacyToken
from spacy.tokens import Doc as spacyDoc
from spacy.tokens import Span as spacySpan

matched_args = 0
matched_args_same_ix = 0
matched_events = 0
matched_events_same_ix = 0

nlp = spacy.load('en_core_web_sm')  # en_core_web_sm 2.0.0


def order_docs_by_topics(docs: Dict[str, Document]) -> Corpus:
    """
    Gets list of document objects and returns a Corpus object.
    The Corpus object contains Document objects which are ordered by their gold
    topics

    The returned Corpus obj has structure likes::

        普通变量是docs本来就有的信息，
        尖括号中的变量是本函数运行后添加的信息。
        <Courpus_obj>
            <Corpus_obj.topics> -> Topic_obj
        <Topic_obj>
            <Topic_obj.docs> -> Document_obj
        Document_obj
            Document_obj.sentences -> Sentence_obj
            gold/pred_event/entity_mentions -> Mention obj
        Sentence_obj
            Sentence_obj.tokens -> Token_obj
        Mention_obj
            cd/wd_coref_chain -> Coref_chain_str
            doc_id -> Document_obj
            sent_id -> Sentence_obj
            tokens -> Token_obj
        Token_obj
            gold_event/entity_cd/wd_coref_chain -> Coref_chain_str

    :param docs: dict of document objects
    :return: Corpus object
    """
    corpus = Corpus()
    for doc_id, doc in docs.items():
        topic_id, doc_no = doc_id.split('_')
        if 'ecbplus' in doc_no:
            topic_id = topic_id + '_' +'ecbplus'
        else:
            topic_id = topic_id + '_' +'ecb'
        if topic_id not in corpus.topics:
            topic = Topic(topic_id)
            corpus.add_topic(topic_id, topic)
        topic = corpus.topics[topic_id]
        topic.add_doc(doc_id, doc)
    return corpus


def load_ECB_plus(processed_ecb_file: str) -> Dict[str, Document]:
    r"""
    This function gets the intermediate data  ECB_Train/test/dev_corpus.text and load it into a dict of Document obj.

    Note: load_ECB_plus 不是说只load 源自X_Xecbplus.xml的数据。
    这里ECB_plus指的是整个语料库。
    参数指定的ECB_Train/Test/Dev_corpus.txt文件中包含整个语料库的信息。

    Example of the text file::

        1_10ecb	0	0	Perennial	-
        1_10ecb	0	1	party	-
        1_10ecb	0	2	girl	-
        1_10ecb	0	3	Tara	HUM16236184328979740
        1_10ecb	0	4	Reid	HUM16236184328979740

    Example of the return::

        {
            '1_10ecb': Document obj,
            '1_11ecb': Document obj,
            ...
        }

    In detail, each Document obj in return dict includes follow info::

        Document_obj
            Document_obj.sentences -> Sentence_obj
        Sentence_obj
            Sentence_obj.tokens -> Token_obj
        Token_obj

    :param processed_ecb_file: The path of the ECB_Train/test/dev_corpus.text.
        e.g. "data/interim/cybulska_setup/ECB_Train_corpus.txt".
    :return: A dictionary of document objects, which represents the documents in the split.
    """
    doc_changed = True
    sent_changed = True
    docs = {}
    last_doc_id = None
    last_sent_id = None

    for line in open(processed_ecb_file, 'r'):
        stripped_line = line.strip()  # 去掉多余的空格
        try:
            # if  line != "\n"
            if stripped_line:
                doc_id, sent_id, token_num, word, coref_chain = stripped_line.split('\t')
                doc_id = doc_id.replace('.xml', '')  # 这句是废话，因为doc_id中都没有“.xml”。
            # if line == "\n"
            else:
                pass
        except:
            # There may be a exception because some special line like:
            # '2_5ecbplus\t0\t9\tAwards\t\tACT16239369414744113'
            # There are 5 \t and you will get 6 elements, instead of 5 elements, after split('\t').
            # The '\t\t' makes a unexpected empty element.
            # So, you need to filter out the empty element.
            row = stripped_line.split('\t')
            clean_row = []
            for item in row:
                # append the normal element
                if item:
                    clean_row.append(item)
                # filter out the empty element
                else:
                    pass
            doc_id, sent_id, token_num, word, coref_chain = clean_row
            doc_id = doc_id.replace('.xml', '')  # 这句是废话，因为doc_id中都没有“.xml”。

        if stripped_line:
            sent_id = int(sent_id)

            # test the change of doc and sent
            if last_doc_id is None:
                last_doc_id = doc_id
            elif last_doc_id != doc_id:
                doc_changed = True
                sent_changed = True
            if last_sent_id is None:
                last_sent_id = sent_id
            elif last_sent_id != sent_id:
                sent_changed = True

            # new Document
            if doc_changed:
                new_doc = Document(doc_id)
                docs[doc_id] = new_doc
                doc_changed = False
                last_doc_id = doc_id

            # new Sentence
            if sent_changed:
                new_sent = Sentence(sent_id)
                new_doc.add_sentence(sent_id, new_sent)
                sent_changed = False
                last_sent_id = sent_id

            # new Token
            new_tok = Token(token_num, word, '-')
            new_sent.add_token(new_tok)

    return docs


def find_args_by_dependency_parsing(dataset: Corpus, is_gold: bool) -> None:
    """
    This function:
        1. Runs dependency parser on the split's sentences,
        2. augments the predicate-argument structures based on the dep parse.

    :param dataset: an object represents the split (Corpus object)
    :param is_gold: whether to match arguments and predicates with gold or predicted mentions
    :return: No return.
        the new predicate-argument structures are add into *dataset*, as what happend in *load_srl_info()*.
    """
    global matched_args, matched_args_same_ix, matched_events,matched_events_same_ix
    matched_args = 0
    matched_args_same_ix = 0
    matched_events = 0
    matched_events_same_ix = 0

    for topic_id, topic in dataset.topics.items():
        for doc_id, doc in topic.docs.items():
            for sent_id, sent in doc.get_sentences().items():
                # dep parse using spaCy
                sent_str = sent.get_raw_sentence()
                parsed_sent = nlp(sent_str)
                # load predicate-argument structures from dep parse
                findSVOs(parsed_sent=parsed_sent, sent=sent, is_gold=is_gold)

    print('matched events : {} '.format(matched_events))
    print('matched args : {} '.format(matched_args))


def find_left_and_right_mentions(dataset, is_gold):
    """
        本函数在一些情况下，把每个event左边的entity设为event的arg0，把event右边的entity设为event的arg1.

        具体来说，对*dataset*中的每个pred/gold_event（取决于is_gold参数），做如下处理：
            1. 如果event没有arg0，找到event左边最近的那个entity，如果这个entity不是event的arg1，amloc和amtmp，那么把这个entity作为event的arg0.
            2. 如果evnet没有arg1，找到event左边最近的那个entity，如果这个entity不是event的arg0，amloc和amtmp，那么把这个entity作为event的arg1.

        :param dataset: an object represents the split (Corpus object)
        :param is_gold: whether to match with gold or predicted mentions
        :return: No return.
            在符合上述具体要求的情况下，
            *sent*[gold/pred_event_mentions][matched_event_index].arg0设为event左边的entity；
            *sent*[gold/pred_event_mentions][matched_event_index].arg1设为event右边的entity.
        """
    for topic_id, topic in dataset.topics.items():
        for doc_id, doc in topic.docs.items():
            for sent_id, sent in doc.get_sentences().items():
                add_left_and_right_mentions(sent, is_gold)


def match_subj_with_event(verb_text, verb_index, subj_text, subj_index, sent, is_gold):
    '''
    Given a verb and a subject extracted by the dependency parser , this function tries to match
    the verb with an event mention and the subject with an entity mention

    :param verb_text: the verb's text
    :param verb_index: the verb index in the parsed sentence
    :param subj_text: the subject's text
    :param subj_index: the subject index in the parsed sentence
    :param sent: an object represents the sentence (Sentence object)
    :param is_gold: whether to match with gold or predicted mentions
    :return: No return.
        *sent*[gold_event_mentions][matched_event_index].arg0/arg1/amloc/amtmp被修改为对应论元
        *sent*[gold_entity_mentions][matched_entity_index].predicates中添加了此entity对应的谓词的id和此论元的类型A0
    '''
    event = match_event(verb_text, verb_index, sent, is_gold)
    if event is not None and event.arg0 is None:
        entity = match_entity(subj_text, subj_index, sent, is_gold)
        if entity is not None:
            if event.arg1 is not None and event.arg1 == (entity.mention_str, entity.mention_id):
                return
            if event.amloc is not None and event.amloc == (entity.mention_str, entity.mention_id):
                return
            if event.amtmp is not None and event.amtmp == (entity.mention_str, entity.mention_id):
                return
            event.arg0 = (entity.mention_str, entity.mention_id)
            entity.add_predicate((event.mention_str, event.mention_id), 'A0')


def match_obj_with_event(verb_text, verb_index, obj_text, obj_index, sent, is_gold):
    '''
    Given a verb and an object extracted by the dependency parser , this function tries to match
    the verb with an event mention and the object with an entity mention
    :param verb_text: the verb's text
    :param verb_index: the verb index in the parsed sentence
    :param obj_text: the object's text
    :param obj_index: the object index in the parsed sentence
    :param sent: an object represents the sentence (Sentence object)
    :param is_gold: whether to match with gold or predicted mentions
    '''
    event = match_event(verb_text, verb_index, sent, is_gold)
    if event is not None and event.arg1 is None:
        entity = match_entity(obj_text, obj_index, sent, is_gold)
        if entity is not None:
            if event.arg0 is not None and event.arg0 == (entity.mention_str, entity.mention_id):
                return
            if event.amloc is not None and event.amloc == (entity.mention_str, entity.mention_id):
                return
            if event.amtmp is not None and event.amtmp == (entity.mention_str, entity.mention_id):
                return
            event.arg1 = (entity.mention_str, entity.mention_id)
            entity.add_predicate((event.mention_str, event.mention_id), 'A1')


def match_event(verb_text, verb_index, sent, is_gold):
    '''
    Given a verb extracted by the dependency parser , this function tries to match
    the verb with an event mention.

    :param verb_text: the verb's text
    :param verb_index: the verb index in the parsed sentence
    :param sent: an object represents the sentence (Sentence object)
    :param is_gold: whether to match with gold or predicted mentions
    :return: the matched event (and None if the verb doesn't match to any event mention)
    '''
    global matched_events, matched_events_same_ix
    sent_events = sent.gold_event_mentions if is_gold else sent.pred_event_mentions
    for event in sent_events:
        event_toks = event.tokens
        for tok in event_toks:
            if tok.get_token() == verb_text:
                if is_gold:
                    matched_events += 1
                elif event.gold_mention_id is not None:
                    matched_events += 1
                if verb_index == int(tok.token_id):
                    matched_events_same_ix += 1
                return event
    return None


def match_entity(entity_text, entity_index, sent, is_gold):
    '''
    Given an argument extracted by the dependency parser , this function tries to match
    the argument with an entity mention.

    :param entity_text: the argument's text
    :param entity_index: the argument index in the parsed sentence
    :param sent: an object represents the sentence (Sentence object)
    :param is_gold: whether to match with gold or predicted mentions
    :return: the matched entity (and None if the argument doesn't match to any event mention)
    '''
    global matched_args, matched_args_same_ix
    sent_entities = sent.gold_entity_mentions if is_gold else sent.pred_entity_mentions
    for entity in sent_entities:
        entity_toks = entity.tokens
        for tok in entity_toks:
            if tok.get_token() == entity_text:
                if is_gold:
                    matched_args += 1
                elif entity.gold_mention_id is not None:
                    matched_args += 1
                if entity_index == int(tok.token_id):
                    matched_args_same_ix += 1
                return entity
    return None

'''
Borrowed with modifications from https://github.com/NSchrading/intro-spacy-nlp/blob/master/subject_object_extraction.py
'''

SUBJECTS = ["nsubj"]  # 主动语态动词的名词型主语
PASS_SUBJ = ["nsubjpass",  "csubjpass"]  # 被动语态动词的名词型主语和短语型主语
OBJECTS = ["dobj", "iobj", "attr", "oprd"]


def getSubsFromConjunctions(subs: list):
    '''
    Finds subjects in conjunctions.

    就是A， B and C。你给出A，我返回[B, C]
    注意ABC是conjunct head，对New York来说，York是head。

    Usage Example::

        >>> import spacy
        >>> nlp = spacy.load('en_core_web_sm')
        >>> txt = "Tom, Jack and Thomas are boys. The two girls are Jenny as well as Diana."
        >>> doc = nlp(txt)
        >>> doc[0]
        Tom
        >>> doc[12]
        Jenny
        >>> getSubsFromConjunctions([doc[0]])
        [Jack, Thomas]
        >>> getSubsFromConjunctions([doc[0], doc[12]])
        [Jack, Thomas, Diana]


    :param subs: found subjects so far
    :return: additional subjects, if exist
    '''
    # 这是Barhom2019原来的算法
    sl1 = getSubsFromConjunctions1(subs)
    # 这是我自己写的基于dep的新算法
    """
    sl2 = getSubsFromConjunctions2(subs)
    """

    # 比较两个算法的不同
    """
    if subs:
        if sl1 != sl2:
            print(subs, ":::", sl1, ":::", sl2, ":::", subs[0].sent)
    """

    # 还是用Barhom的原来算法
    return sl1

def getSubsFromConjunctions1(subs: list):
    moreSubs = []
    for sub in subs:
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreSubs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
            if len(moreSubs) > 0:
                moreSubs.extend(getSubsFromConjunctions(moreSubs))

    return moreSubs

def getSubsFromConjunctions2(sub_list: List) -> List:
    extended_sub_set = set()

    for sub in sub_list:
        rights = list(sub.rights)
        for right in rights:
            if right.dep_ == "conj" and right.head == sub:
                if right not in sub_list:
                    extended_sub_set.add(right)
                extended_sub_set = extended_sub_set | set(getSubsFromConjunctions2([right]))

    extended_sub_list = list(extended_sub_set)

    def get_index(e):
        return e.i
    extended_sub_list.sort(key=get_index)

    return extended_sub_list


def getObjsFromConjunctions(objs):
    '''
    Finds objects in conjunctions (and)
    :param objs: found objects so far
    :return: additional objects, if exist
    '''
    moreObjs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreObjs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"])
            if len(moreObjs) > 0:
                moreObjs.extend(getObjsFromConjunctions(moreObjs))
    return moreObjs


def getObjsFromPrepositions(deps):
    '''
    Finds objects in prepositions

    :param deps: dependencies extracted by spaCy parser
    :return: objects extracted from prepositions
    '''
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" and dep.dep_ == "prep":
            if [tok for tok in dep.rights if tok.dep_ in OBJECTS]:
                print(1)
            objs.extend([tok for tok in dep.rights if tok.dep_ in OBJECTS])
    return objs


def getObjFromXComp(deps):
    '''
     Finds objects in XComp phrases (X think that [...])

    :param deps: dependencies extracted by spaCy parser
    :return: objects extracted from XComp phrases
    '''
    for dep in deps:
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(getObjsFromPrepositions(rights))
            if len(objs) > 0:
                return v, objs
    return None, None


def getAllSubs(v: spacyToken) -> Tuple[List[spacyToken], List[spacyToken]]:
    """
    Finds all possible subjects of an extracted verb.

    :param v: an extracted verb
    :return: A tuple that includes all possible subjects of the verb.
        First element in the tuple is a list of 给定动词（假设是主动语态）的主语.
        Second element in the tuple is a list of 给定动词(假设是被动语态)的主语.
    """
    # 给定动词（假设是主动语态）的主语
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    """
    为什么要刨除DET呢？是为了排除类似如下的情况：
        代指：
            "that" in "That is a good idea".
        省略：
            "some" in "Some is dressed in drag."（完整应为some one）
            "another" in "two people have been killed and another wounded in a shooting."（完整应为another people）
    """
    if len(subs) > 0:
        subs.extend(getSubsFromConjunctions(subs))

    # 给定动词(假设是被动语态)的主语
    pass_subs = [tok for tok in v.lefts if tok.dep_ in PASS_SUBJ and tok.pos_ != "DET"]
    """
    为什么不扩展并列结构？
    if len(pass_subs) > 0:
        pass_subs.extend(getSubsFromConjunctions(pass_subs))
    """

    #
    return subs, pass_subs


def getAllObjs(v: spacyToken):
    '''
     Finds all the objects of an extracted verb

    :param v: an extracted verb
    :return: all possible objects of the verb
    '''
    rights = list(v.rights)
    for i in rights:
        if i.dep_ == "oprd":
            print("[", i, "]", i.sent)
    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
    """
    objs1 = [tok for tok in rights if tok.dep_ in OBJECTS and tok.pos_ != "DET"]
    if objs != objs1:
        print("objs:", objs, "objs1:", objs1, "[", v.sent, "]")
    objs: [any] objs1: [] [ After telling People in October that she did n't `` need to do any of that anymore , '' `` American Pie '' actress and former travel reporter Tara Reid checked into Promises Treatment Center on Friday for an unspecified problem . ]
    objs: [that] objs1: [] [ Los Angeles general manager Tony Reagins has sounded like a broken record saying that resigning Teixeira was his top priority , and backed that up now with an eight-year offer . ]
    objs: [all] objs1: [] [ Now , after guiding the Colts back to the playoffs - and breaking Cam Newton's single - season passing record for a first - year player - it's safe to say Luck exceeded just about all of them . ]
    objs: [all] objs1: [] [ Now , after guiding the Colts back to the playoffs - and breaking Cam Newton's single - season passing record for a first - year player - it's safe to say Luck exceeded just about all of them . ]
    objs: [all] objs1: [] [ Now , after guiding the Colts back to the playoffs - and breaking Cam Newton 's single - season passing record for a first - year player - it's safe to say Luck exceeded just about all of them . ]
    objs: [both] objs1: [] [ A fired accountant bought a shotgun the day after he lost his job , then returned to the office the following week and opened fire on his bosses , wounding both and killing a receptionist , the Associated Press reported . ]
    objs: [that] objs1: [] [ The first of the deaths this weekend was that of a New Zealand climber who fell on Friday morning . ]
    objs: [all] objs1: [] [ While Apple 's final keynote address at the annual MacWorld convention in San Francisco did n't contain the iPhone-related announcements many had hoped for , we were given all of the details on Apple 's latest revamp to their largest and most powerful MacBook Pro . ]
    objs: [a] objs1: [] [ The temblor struck at 2 : 09 a . m . and was centered in Geyserville , about 20 miles north of Santa Rosa , according to the U . ]
    """
    objs.extend(getObjsFromPrepositions(rights))

    potentialNewVerb, potentialNewObjs = getObjFromXComp(rights)
    if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
        objs.extend(potentialNewObjs)
        v = potentialNewVerb
    if len(objs) > 0:
        objs.extend(getObjsFromConjunctions(objs))
    return objs


def findSVOs(parsed_sent: spacyDoc, sent: Sentence, is_gold: bool) -> None:
    """
    Given the spaCy dep parse result of a sentences (the Doc obj), the function:
        1. extracts its verbs, their subjects and objects,
        2. matches the verbs with event mentions
        3. matches the subjects and objects with entity mentions, and set them
           as Arg0 and Arg1 respectively.
        4. finds nominal event mentions (名词化的动词) with possesors, matches the
           possesor with entity mention and set it as Arg0.

    :param parsed_sent: a Doc obj, which includes spaCy dep parse result of a sentences (the Doc obj).
    :param sent: a Sentence object corresponding to the sentence in *parsed_sent*
    :param is_gold: whether to match with gold or predicted mentions
    :return: No return.
        the new predicate-argument structures are add into *sent*, as what happend in *load_srl_info()*.
    """
    global matched_events, matched_events_same_ix
    global matched_args, matched_args_same_ix

    # 找出所有实意动词（动词中刨去助动词，aux就是助动词）
    """
    本来要执行：
    verbs = [tok for tok in parsed_sent if tok.pos_ == "VERB"]
    因为实意动词标为VERB，助动词标为AUX，参见标签集。
    但是因为bug：https://github.com/explosion/spaCy/issues/593
    AUX标签从不出现，所有动词都是VERB。
    所以需要手动补救。
    """
    verbs = [tok for tok in parsed_sent if tok.pos_ == "VERB" and tok.dep_ != "aux"]

    for v in verbs:
        #
        subs, pass_subs = getAllSubs(v)
        objs = getAllObjs(v)
        #
        if len(subs) > 0 or len(objs) > 0 or len(pass_subs) > 0:
            for sub in subs:
                match_subj_with_event(verb_text=v.orth_,
                                      verb_index=v.i, subj_text=sub.orth_,
                                      subj_index=sub.i, sent=sent, is_gold=is_gold)

            for obj in objs:
                match_obj_with_event(verb_text=v.orth_,
                                        verb_index=v.i, obj_text=obj.orth_,
                                        obj_index=obj.i, sent=sent, is_gold=is_gold)
            for obj in pass_subs:
                match_obj_with_event(verb_text=v.orth_,
                                        verb_index=v.i, obj_text=obj.orth_,
                                        obj_index=obj.i, sent=sent, is_gold=is_gold)

    find_nominalizations_args(parsed_sent, sent, is_gold) # Handling nominalizations


def find_nominalizations_args(parsed_sent, sent, is_gold):
    '''
    The function finds nominal event mentions with possesors, matches the possesor
    with entity mention and set it as Arg0.
    :param parsed_sent: a sentence, parsed by spaCy
    :param sent: the original Sentence object
    :param is_gold: whether to match with gold or predicted mentions
    '''
    possible_noms = [tok for tok in parsed_sent if tok.pos_ == "NOUN"]
    POSS = ['poss', 'possessive']
    for n in possible_noms:
        subs = [tok for tok in n.lefts if tok.dep_ in POSS and tok.pos_ != "DET"]
        if len(subs) > 0:
            for sub in subs:
                match_subj_with_event(verb_text=n.orth_,
                                      verb_index=n.i, subj_text=sub.orth_,
                                      subj_index=sub.i, sent=sent, is_gold=is_gold)


def add_left_and_right_mentions(sent, is_gold):
    """
    本函数在一些情况下，把每个event左边的entity设为event的arg0，把event右边的entity设为event的arg1.

    具体来说，对一个sent中的每个pred/gold_event（取决于is_gold参数），做如下处理：
        1. 如果event没有arg0，找到event左边最近的那个entity，如果这个entity不是event的arg1，amloc和amtmp，那么把这个entity作为event的arg0.
        2. 如果event没有arg1，找到event右边最近的那个entity，如果这个entity不是event的arg0，amloc和amtmp，那么把这个entity作为event的arg1.

    :param sent: Sentence object
    :param is_gold: whether to match with gold or predicted mentions
    :return: No return.
        在符合上述具体要求的情况下，
        *sent*[gold/pred_event_mentions][matched_event_index].arg0设为event左边的entity；
        *sent*[gold/pred_event_mentions][matched_event_index].arg1设为event右边的entity.
    """
    sent_events = sent.gold_event_mentions if is_gold else sent.pred_event_mentions
    for event in sent_events:
        if event.arg0 is None:
            left_ent = sent.find_nearest_entity_mention(event, is_left=True, is_gold=is_gold)
            if left_ent is not None:
                double_arg = False
                if event.arg1 is not None and event.arg1 == (left_ent.mention_str, left_ent.mention_id):
                    double_arg = True
                if event.amloc is not None and event.amloc == (left_ent.mention_str, left_ent.mention_id):
                    double_arg = True
                if event.amtmp is not None and event.amtmp == (left_ent.mention_str, left_ent.mention_id):
                    double_arg = True

                if not double_arg:
                    event.arg0 = (left_ent.mention_str, left_ent.mention_id)
                    left_ent.add_predicate((event.mention_str, event.mention_id), 'A0')

        if event.arg1 is None:
            right_ent = sent.find_nearest_entity_mention(event, is_left=False, is_gold=is_gold)
            if right_ent is not None:
                double_arg = False
                if event.arg0 is not None and event.arg0 == (right_ent.mention_str, right_ent.mention_id):
                    double_arg = True
                if event.amloc is not None and event.amloc == (right_ent.mention_str, right_ent.mention_id):
                    double_arg = True
                if event.amtmp is not None and event.amtmp == (right_ent.mention_str, right_ent.mention_id):
                    double_arg = True
                if not double_arg:
                    event.arg1 = (right_ent.mention_str, right_ent.mention_id)
                    right_ent.add_predicate((event.mention_str, event.mention_id), 'A1')