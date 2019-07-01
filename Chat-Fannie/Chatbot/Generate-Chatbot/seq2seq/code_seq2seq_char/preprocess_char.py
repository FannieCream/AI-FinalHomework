"""
把 chicken_and_gossip数据 文件格式转换为可训练格式
Code from: QHDuan  url: https://github.com/qhduan/just_another_seq2seq

"""

from config.path_config import chicken_and_gossip_path
from config.path_config import chatbot_data_cg_char_dir
from config.path_config import chatbot_data_cg_ws_anti
from config.path_config import chatbot_data_cg_xy_anti
from config.path_config import model_ckpt_cg_anti

from utils.model_util.word_sequence import WordSequence
from utils.text_preprocess import txtRead
from tqdm import tqdm
import pickle
import sys
import re

sys.path.append('..')


def make_split(line):
    """
    构造合并两个句子之间的符号
    """
    if re.match(r'.*([，。…？！～\.,!?])$', ''.join(line)):
        return []
    return ['，']


def good_line(line):
    if len(re.findall(r'[a-zA-Z0-9]', ''.join(line))) > 2:
        return False
    return True


def regular(sen, limit=50):
    sen = re.sub(r'\.{3,100}', '…', sen)
    sen = re.sub(r'…{2,100}', '…', sen)
    sen = re.sub(r'[,]{1,100}', '，', sen)
    sen = re.sub(r'[\.]{1,100}', '。', sen)
    sen = re.sub(r'[\?]{1,100}', '？', sen)
    sen = re.sub(r'[!]{1,100}', '！', sen)
    if len(sen) > limit:
        sen = sen[0:limit]
    return sen


def creat_train_data_of_cg_corpus(limit=50, x_limit=2, y_limit=2):
    x_datas = []
    y_datas = []
    max_len = 0
    sim_datas = txtRead(chicken_and_gossip_path, encodeType="utf-8")
    for sim_datas_one in sim_datas[1:]:
        if sim_datas_one:
            sim_datas_one_split = sim_datas_one.strip().split("\t")
            if len(sim_datas_one_split) == 2:
                len_x1 = len(sim_datas_one_split[0])
                len_x2 = len(sim_datas_one_split[1])
                max_len = max(len_x1, len_x2, max_len)

                sentence_org = regular(sim_datas_one_split[0], limit=limit)
                sentence_sim = regular(sim_datas_one_split[1], limit=limit)
                x_datas.append([sen for sen in sentence_org])
                y_datas.append([sen for sen in sentence_sim])

    datas = list(zip(x_datas, y_datas))
    datas = [
        (x, y)
        for x, y in datas
        if len(x) < limit and len(y) < limit and len(y) >= y_limit and len(x) >= x_limit
    ]
    x_datas, y_datas = zip(*datas)

    print(' Start Fit WordSequence ！')

    ws_input = WordSequence()
    ws_input.fit(x_datas + y_datas)
    print(' Fit WordSequence Done ！')

    print('Begin Dump !')
    pickle.dump((x_datas, y_datas),open(chatbot_data_cg_xy_anti, 'wb'))
    pickle.dump(ws_input, open(chatbot_data_cg_ws_anti, 'wb'))

    print(' Dump Done ! ')
    print(max_len)


if __name__ == '__main__':
    creat_train_data_of_cg_corpus()
