from __future__ import print_function
from utils.text_preprocess import txtRead, txtWrite
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
import multiprocessing
import logging
import sys
import os

def train_word2vec_by_word():
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logging.info("running")

    inp = "Y:/BaiduNetdiskDownload/cut_zhwiki_wiki_parse/cut_zhwiki_wiki_parse.txt"
    outp1 = "w2v_model_wiki.model"
    outp2 = "w2v_model_wiki_word.vec"
    model = Word2Vec(LineSentence(inp), size=300, window=5, min_count=5, workers=multiprocessing.cpu_count())
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)

def train_word2vec_by_char():
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logging.info("running")

    inp = "Y:/BaiduNetdiskDownload/cut_zhwiki_wiki_parse/cut_zhwiki_wiki_parse_char.txt"
    outp1 = "w2v_model_wiki.model"
    outp2 = "w2v_model_wiki_char.vec"
    model = Word2Vec(LineSentence(inp), size=300, window=5, min_count=5, workers=multiprocessing.cpu_count())
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)


if __name__ == '__main__':
    train_word2vec_by_word()
