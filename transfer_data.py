#!/usr/bin/env python3
# coding: utf-8
# File: transfer_data.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-12-24

import os
from collections import Counter

class TransferData:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.label_dict = {
                      '检查和检验': 'CHECK',
                      '症状和体征': 'SIGNS',
                      '疾病和诊断': 'DISEASE',
                      '治疗': 'TREATMENT',
                      '身体部位': 'BODY'}

        self.cate_dict ={
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
        self.origin_path = os.path.join(cur, 'data_origin')
        self.train_filepath = os.path.join(cur, 'train.txt')
        return


    def transfer(self):
        f = open(self.train_filepath, 'w+')
        count = 0
        for root,dirs,files in os.walk(self.origin_path):
            for file in files:
                filepath = os.path.join(root, file)
                if 'original' not in filepath:
                    continue
                label_filepath = filepath.replace('.txtoriginal','')
                print(filepath, '\t\t', label_filepath)
                content = open(filepath).read().strip()
                res_dict = {}
                for line in open(label_filepath):
                    res = line.strip().split('	')
                    start = int(res[1])
                    end = int(res[2])
                    label = res[3]
                    label_id = self.label_dict.get(label)
                    for i in range(start, end+1):
                        if i == start:
                            label_cate = label_id + '-B'
                        else:
                            label_cate = label_id + '-I'
                        res_dict[i] = label_cate

                for indx, char in enumerate(content):
                    char_label = res_dict.get(indx, 'O')
                    print(char, char_label)
                    f.write(char + '\t' + char_label + '\n')
        f.close()
        return



if __name__ == '__main__':
    handler = TransferData()
    train_datas = handler.transfer()