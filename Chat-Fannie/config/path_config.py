# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @time     :2019/6/16
# @author   :FannieCream
# @function :

import pathlib
import sys
import os

# base dir
projectdir = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)
sys.path.append(projectdir)
print(projectdir)

path_params=projectdir + '/config/params.json'
# corpus
chicken_and_gossip_path = projectdir + '/data/corpus/chicken_and_gossip.txt'

# chatbot data char
chatbot_data_cg_char_dir = projectdir + '/Chatbot/Generate-Chatbot/seq2seq/data_seq2seq_char'
chatbot_data_cg_ws_anti=projectdir + '/ChatBot/Generate-Chatbot/seq2seq/data_seq2seq_char/train_data_web_ws_anti.pkl'
chatbot_data_cg_xy_anti=projectdir + '/ChatBot/Generate-Chatbot/seq2seq/data_seq2seq_char/train_data_web_xy_anti.pkl'
model_ckpt_cg_anti=projectdir + '/ChatBot/Generate-Chatbot/seq2seq/model_seq2seq_char/model_ckpt_char_cg.ckp'

