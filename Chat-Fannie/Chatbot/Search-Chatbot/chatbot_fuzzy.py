# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/10
# @author   :FannieCream
# @function :

from config.path_config import chicken_and_gossip_path
from utils.text_preprocess import txtRead, txtWrite
from config.path_config import projectdir
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import pickle
import time
import re


def count_same_char(x1, x2):
    '''获取相同字符的个数'''
    res = []
    for x in x1:
      if x in x2:
        res.append(x)
    if res:
        return len(res)
    else:
        return 0

def fuzzy_fuzzywuzzy_list(fuzz, user_input, qa_list, collection, topn=50):
    '''编辑距离，速度比较慢，比起匹配方法，能够处理字符不一样的问题'''

    user_input_set = [user_input_one for user_input_one in user_input]

    same_char_list = []
    max_data = 0
    max_data_list = []
    count_collection_new_one = 0
    for collection_new_one in collection: # 获取相同字符串多的问题
        count_same_char_one = len([x for x in user_input_set if x in collection_new_one])

        if count_same_char_one > 0:
            same_char_list.append((count_collection_new_one, count_same_char_one))
        if count_same_char_one > max_data:
            max_data_list.append(count_same_char_one)
            max_data = count_same_char_one
        count_collection_new_one += 1

    list_max_count = []
    len_max_data_list = len(max_data_list)
    for x in range(len_max_data_list):  # 获取前20排名
        for k,l in same_char_list:
            if l == max_data_list[len_max_data_list -1 - x]:
                list_max_count.append(qa_list[k]) #问答重这里取出来
        if len(list_max_count) >= 5000:
            list_max_count = list_max_count[0:5000]
            break

    result =  process.extract(user_input, list_max_count, scorer=fuzz.token_set_ratio, limit=topn)
    return result



if __name__ == '__main__':
    qa_list = txtRead(chicken_and_gossip_path)
    questions = [qa.strip().split("\t")[0] for qa in qa_list]
    print("-----------------------------  ENJOYING YOUT CHATTING --------------------------")
    sen = "Hello World!"

    list_fuzzyfinder = fuzzy_fuzzywuzzy_list(fuzz, sen, qa_list, questions, topn=5)
    print("菜菜 SAY ： " + sen)
    print("菜菜 SAY ： 快来和我聊天吧~" )

    while True:
        print("YOU SAY : ")
        ques = input()
        if ques in ('exit', 'quit'):
            exit(0)
        list_fuzzyfinder = fuzzy_fuzzywuzzy_list(fuzz, ques, qa_list, questions, topn=5)
        print("菜菜 SAY ： " + list_fuzzyfinder[0][0].split("\t")[1].strip())
        print("推荐结果: ")
        print(list_fuzzyfinder)
