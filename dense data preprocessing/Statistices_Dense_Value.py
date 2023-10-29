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
    description="Statistics of Kaggle Dataset Dense Value (Max, Min, Mean, Variance, Unknown click......)"
)
parser.add_argument("--fea-count-path", type=str, default="/work/twsugkm569/data/kaggle/statistic/kaggle_dense_fea_count.pkl")
parser.add_argument("--save-path", type=str, default="/work/twsugkm569/data/kaggle/statistic/")
parser.add_argument("--file-name", type=str, default="statistics_dense_fea.pkl")

args = parser.parse_args()
# -

with open(args.fea_count_path, "rb") as file:
    test = pickle.load(file)

dic = [{'total_sum':0,'total_square_sum':0,'total_click':0,'unknown':0,'mean':0,'variance':0,'min':0,'max':0} for i in range(13)]

for i in range(13):
    total_square_sum = 0
    total_sum = 0
    total_click = 0
    unknown = 0
    minimum = 999
    maximum = -999
    for keys, values in test[i].items():
        if keys != '':
            total_sum += int(keys) * int(values)
            total_square_sum += pow(int(keys),2) * int(values)
            total_click += int(values)
            if int(keys) < minimum:
                minimum = int(keys)
            if int(keys) > maximum:
                maximum = int(keys)
        else:
            unknown = values
    mean = total_sum/total_click
    variance = (total_square_sum/total_click) - pow(mean,2)
    
    dic[i]['total_sum'] = total_sum
    dic[i]['total_square_sum'] = total_square_sum
    dic[i]['total_click'] = total_click
    dic[i]['unknown'] = unknown
    dic[i]['mean'] = mean
    dic[i]['variance'] = variance
    dic[i]['min'] = minimum
    dic[i]['max'] = maximum
    
    print(f"The {i}th field dense feature statistics:")
    print("=========================================")
    print("total_sum: ",total_sum)
    print("total_click: ",total_click)
    print("unknown: ",unknown)
    print("mean: ",mean)
    print("variance: ",variance)
    print("min: ",minimum)
    print("max: ",maximum)
    print("#########################################")

with open(args.save_path + args.file_name, "wb") as file:
    pickle.dump(dic,file)
print("*************")
print("SAVE FINISH!!")
print("*************")
