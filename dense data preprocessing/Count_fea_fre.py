# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

ori_train_data = "/work/twsugkm569/data/kaggle_before_preprocess/train.txt"
split_train_data = "/work/twsugkm569/data/kaggle_before_preprocess/train_s.txt"
split_valid_data = "/work/twsugkm569/data/kaggle_before_preprocess/valid_s.txt"
train_data_csv = "/work/twsugkm569/data/kaggle_before_preprocess/train.csv"
valid_data_csv = "/work/twsugkm569/data/kaggle_before_preprocess/valid.csv"

import random, math, os
import matplotlib.pyplot as plt
import math
import pickle
random.seed(0)


# https://github.com/WayneDW/AutoInt/blob/master/Dataprocess/Criteo/scale.py
def scale(x, i):
    mean = [3.5024133170754044
            ,105.84841979766546
            ,26.913041020611274
            ,7.322680248873305
            ,18538.991664871523
            ,116.06185085211598
            ,16.333130032135028
            ,12.517042137556713
            ,106.1098234380509
            ,0.6175294977722137
            ,2.7328343170173044
            ,0.9910356287721244
            ,8.217461161174054 ]
    std = [9.429076218877869
            ,391.4578184172999
            ,397.97257749551636
            ,8.793230590188488
            ,69394.60106925869
            ,382.5664439963476
            ,66.04975449211732
            ,16.688884385665453
            ,220.28309147514412
            ,0.6840505417423962
            ,5.1990708255393745
            ,5.597723612336433
            ,16.21193233240115 ]
    if x == '':
        return '0'
    else:
        x = float(x) - mean[i - 1]
        x = x / std[i - 1]
        return str(x)


def cnt_freq_train(inputs):
    count_freq = []
    for i in range(40):
        count_freq.append({})
    for idx, line in enumerate(open(inputs)):
        line = line.replace('\n', '').split('\t')
        if idx % 1000000 == 0 and idx > 0:
            print(idx)
        for i in range(1, 40):
            if i < 14:
                #line[i] = project_numeric(line[i])
                line[i] = scale(line[i], i)
            if line[i] not in count_freq[i]:
                count_freq[i][line[i]] = 0
            count_freq[i][line[i]] += 1
    return count_freq


print('Count the frequency~')
freq_dict = cnt_freq_train(ori_train_data)
print('Count the frequency finish!')

with open("fea_count.pkl", "wb") as file:
    pickle.dump(freq_dict,file)
print('Save finish!')

import pickle
with open("fea_count.pkl","rb") as file:
    x = pickle.load(file)
    print(x)


