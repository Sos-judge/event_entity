def word_is_name(word):
    """
    1.mention_str含有数字或"."，则不是人名
    2.mention_str的首个字符若是字母，则可能是人名转3；若不是字母，则不是人名.
    3.mention_str含有一个单词：若不在英文词典，则是人名；若在英文词典，则可能是->查常用姓名表，若在则是，不在则不是。
    4.mention_str一个以上单词：整体在英文词典，则不是；整体不在英文词典，则可能是->取str的前两个单词，如果这两个单词在词典内都搜不到，则是人名；
      如果这两个单词中的任意一个在词典中搜到了->查询第一个单词是否在first_name表，第二个单词是否在last_name表，有一个在则是人名；反之，不是人名。

    缺陷：例如Andrew Luck / Andrew 会被认定为人名， Luck不会被认定为人名
    """
    import re
    import json

    if bool(re.search(r'\d', word)): # 如果字符串含有数字 不是人名
        return False
    if '.' in word:
        return False

    with open("lib/human_name_tools/data/english_dictionary.json", 'r') as f:
        en_dic = json.load(f)
    with open("lib/human_name_tools/data/first_names.all.txt", "r") as file_frist_name:
        first_name_list = file_frist_name.read().splitlines()
    with open("lib/human_name_tools/data/last_names.all.txt", "r") as file_last_name:
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