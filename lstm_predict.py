#!/usr/bin/env python3
# coding: utf-8
# File: lstm_predict.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-5-23

import numpy as np
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,load_model
from keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
from keras_contrib.layers.crf import CRF
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class LSTMNER:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.train_path = os.path.join(cur, 'data/train.txt')
        self.vocab_path = os.path.join(cur, 'model/vocab.txt')
        self.embedding_file = os.path.join(cur, 'model/token_vec_300.bin')
        self.model_path = os.path.join(cur, 'model/tokenvec_bilstm2_crf_model_20.h5')
        self.word_dict = self.load_worddict()
        self.class_dict ={
                         'O':0,
                         'TREATMENT-I': 1,
                         'TREATMENT-B': 2,
                         'BODY-B': 3,
                         'BODY-I': 4,
                         'SIGNS-I': 5,
                         'SIGNS-B': 6,
                         'CHECK-B': 7,
                         'CHECK-I': 8,
                         'DISEASE-I': 9,
                         'DISEASE-B': 10
                        }
        self.label_dict = {j:i for i,j in self.class_dict.items()}
        self.EMBEDDING_DIM = 300
        self.EPOCHS = 10
        self.BATCH_SIZE = 128
        self.NUM_CLASSES = len(self.class_dict)
        self.VOCAB_SIZE = len(self.word_dict)
        self.TIME_STAMPS = 150
        self.embedding_matrix = self.build_embedding_matrix()
        self.model = self.tokenvec_bilstm2_crf_model()
        self.model.load_weights(self.model_path)

    '加载词表'
    def load_worddict(self):
        vocabs = [line.strip() for line in open(self.vocab_path)]
        word_dict = {wd: index for index, wd in enumerate(vocabs)}
        return word_dict

    '''构造输入，转换成所需形式'''
    def build_input(self, text):
        x = []
        for char in text:
            if char not in self.word_dict:
                char = 'UNK'
            x.append(self.word_dict.get(char))
        x = pad_sequences([x], self.TIME_STAMPS)
        return x

    def predict(self, text):
        str = self.build_input(text)
        raw = self.model.predict(str)[0][-self.TIME_STAMPS:]
        result = [np.argmax(row) for row in raw]
        chars = [i for i in text]
        tags = [self.label_dict[i] for i in result][len(result)-len(text):]
        res = list(zip(chars, tags))
        print(res)
        return res

    '''加载预训练词向量'''
    def load_pretrained_embedding(self):
        embeddings_dict = {}
        with open(self.embedding_file, 'r') as f:
            for line in f:
                values = line.strip().split(' ')
                if len(values) < 300:
                    continue
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_dict[word] = coefs
        print('Found %s word vectors.' % len(embeddings_dict))
        return embeddings_dict

    '''加载词向量矩阵'''
    def build_embedding_matrix(self):
        embedding_dict = self.load_pretrained_embedding()
        embedding_matrix = np.zeros((self.VOCAB_SIZE + 1, self.EMBEDDING_DIM))
        for word, i in self.word_dict.items():
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix

    '''使用预训练向量进行模型训练'''
    def tokenvec_bilstm2_crf_model(self):
        model = Sequential()
        embedding_layer = Embedding(self.VOCAB_SIZE + 1,
                                    self.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.TIME_STAMPS,
                                    trainable=False,
                                    mask_zero=True)
        model.add(embedding_layer)
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(self.NUM_CLASSES)))
        crf_layer = CRF(self.NUM_CLASSES, sparse_target=True)
        model.add(crf_layer)
        model.compile('adam', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])
        model.summary()
        return model

if __name__ == '__main__':
    ner = LSTMNER()
    while 1:
        s = input('enter an sent:').strip()
        ner.predict(s)

