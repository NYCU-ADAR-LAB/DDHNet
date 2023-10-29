# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pickle
import argparse

# +
parser = argparse.ArgumentParser(
    description="Caculate Kaggle Dense Value Distribution"
)
parser.add_argument("--data-path", type=str, default="/work/twsugkm569/data/kaggle/dataset_before_preprocess")
parser.add_argument("--save-path", type=str, default="/work/twsugkm569/data/kaggle/statistic/")
parser.add_argument("--file-name", type=str, default="kaggle_dense_fea_count.pkl")

args = parser.parse_args()

# +
#train_txt = args.data_path + "/train.txt"
#train.txt ==> Original Dataset without any dense category preprocess

train_txt = args.data_path + "/train.txt" 
test_txt = args.data_path + "/test.txt"   
readme = args.data_path + "/readme.txt"
# -

dic = [{} for i in range(13)]

with open(train_txt) as f:
    for i, line in enumerate(f):
        temp = line.split('\t')
        del temp[0]
        for j in range(0,13):
            if temp[j] in dic[j].keys():
                dic[j][temp[j]] += 1
            else:
                dic[j][temp[j]] = 1
        if i % 10000 == 0:
            print("Finish {:9} Samples of Dataset".format(i))

with open(args.save_path + args.file_name, "wb") as file:
    pickle.dump(dic,file)


