#!/usr/bin/env python3
# coding: utf-8
# File: lstm_train.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-12-24

import numpy as np
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
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
        self.datas, self.word_dict = self.build_data()
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
        self.EMBEDDING_DIM = 300
        self.EPOCHS = 20
        self.BATCH_SIZE = 128
        self.NUM_CLASSES = len(self.class_dict)
        self.VOCAB_SIZE = len(self.word_dict)
        self.TIME_STAMPS = 150
        self.embedding_matrix = self.build_embedding_matrix()

    '''构造数据集'''
    def build_data(self):
        datas = []
        sample_x = []
        sample_y = []
        vocabs = {'UNK'}
        for line in open(self.train_path):
            line = line.rstrip().split('\t')
            if not line:
                continue
            char = line[0]
            if not char:
                continue
            cate = line[-1]
            sample_x.append(char)
            sample_y.append(cate)
            vocabs.add(char)
            if char in ['。','?','!','！','？']:
                datas.append([sample_x, sample_y])
                sample_x = []
                sample_y = []
        word_dict = {wd:index for index, wd in enumerate(list(vocabs))}
        self.write_file(list(vocabs), self.vocab_path)
        return datas, word_dict

    '''将数据转换成keras所需的格式'''
    def modify_data(self):
        x_train = [[self.word_dict[char] for char in data[0]] for data in self.datas]
        y_train = [[self.class_dict[label] for label in data[1]] for data in self.datas]
        x_train = pad_sequences(x_train, self.TIME_STAMPS)
        y = pad_sequences(y_train, self.TIME_STAMPS)
        y_train = np.expand_dims(y, 2)
        return x_train, y_train

    '''保存字典文件'''
    def write_file(self, wordlist, filepath):
        with open(filepath, 'w+') as f:
            f.write('\n'.join(wordlist))

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

    '''训练模型'''
    def train_model(self):
        x_train, y_train = self.modify_data()
        model = self.tokenvec_bilstm2_crf_model()
        history = model.fit(x_train[:], y_train[:], batch_size=self.BATCH_SIZE, epochs=self.EPOCHS)
        self.draw_train(history)
        model.save(self.model_path)
        return model

    '''绘制训练曲线'''
    def draw_train(self, history):
        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.show()
        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.show()
        # 7836/7836 [==============================] - 205s 26ms/step - loss: 17.1782 - acc: 0.9624

if __name__ == '__main__':
    ner = LSTMNER()
    ner.train_model()