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
parser.add_argument("--fea-statistic-path", type=str, default="/work/twsugkm569/data/kaggle/statistic/statistics_dense_fea.pkl")
parser.add_argument("--fea-count-path", type=str, default="/work/twsugkm569/data/kaggle/statistic/kaggle_dense_fea_count.pkl")
parser.add_argument('--clamp', action='store_true', default=False)
parser.add_argument("--save-path", type=str, default="/work/twsugkm569/data/kaggle/statistic/")
parser.add_argument('--save', action='store_true', default=False)
parser.add_argument("--percentile", type=float, default=1.0)

args = parser.parse_args()

# +
#import pickle
#with open("/work/twsugkm569/data/kaggle_gary/statistics_dense_fea.pkl","rb") as file:
#    x = pickle.load(file)
#    for i in x:
#        print(i)
# -

#Load original data
with open(args.fea_statistic_path, "rb") as file:
    total_data = pickle.load(file)
with open(args.fea_count_path, "rb") as file:
    fea_count = pickle.load(file)


def str2int(data = None, sort = False):
    if '' in data.keys(): del data['']
    #data_int = [int(i) for i in data.keys()]
    data_pro = {}
    for keys, values in data.items():
        data_pro[int(keys)] = values
    if sort:
        data_pro = dict(sorted(data_pro.items()))
    
    data_fea_type = [i for i in data_pro.keys()]
    data_fea_count = [i for i in data_pro.values()]
    return data_fea_type, data_fea_count


def get_field_inform(total_data = None, field_num = 0, percentile = False):
    print('\033[0m' + "#########################################")
    if percentile is False:
        print('\033[91m' + f"The {field_num} th field dense feature statistics")
    else:
        print('\033[91m' + f"The {field_num} th field dense feature statistics" +'\033[96m' + ' -> AFTER PERCENTILE')
    print('\033[0m' + "=========================================")
    print("total_sum: ",total_data[field_num]['total_sum'])
    print("total_square_sum: ",total_data[field_num]['total_square_sum'])
    print("total_click: ",total_data[field_num]['total_click'])
    print("unknown: ",total_data[field_num]['unknown'])
    print("mean: ",total_data[field_num]['mean'])
    print("variance: ",total_data[field_num]['variance'])
    print("min: ",total_data[field_num]['min'])
    print("max: ",total_data[field_num]['max'])
    print("#########################################")
    
    return total_data[field_num]['max']


print("================================")
print('\033[96m' + f"       Percentile: {args.percentile}")
print('\033[0m' + "================================")

dic = [{} for i in range(13)]
inform_list = [{} for i in range(13)]
for i in range(13):
    print('\033[45m' + f"@@ {i}")
    #####################
    fea_list = []
    total_square_sum = 0
    total_sum = 0
    total_click = 0
    unknown = 0
    minimum = 999
    maximum = -999
    old_max = 0
    new_max = 0
    #####################
    old_max = get_field_inform(total_data = total_data, field_num = i)
    data_fea_type, data_fea_count = str2int(data = fea_count[i], sort = True)
    #flatten all feature
    #for j in range(len(data_fea_type)):
    #    for k in range(data_fea_count[j]):
    #        fea_list.append(data_fea_type[j])
    for ind,fea_type in enumerate(data_fea_type):
        fea_temp = [fea_type for i in range(data_fea_count[ind])]
        fea_list.extend(fea_temp)
    #percentile
    length = len(fea_list)
    elimate_length = int(length * (1 - args.percentile))
    print("Original feature count: ", len(fea_list))
    fea_list = fea_list[:-elimate_length]
    if args.clamp:
        for j in range(int(length * (1 - args.percentile))):
            fea_list.append(fea_list[-1])
    print("Percentile feature count: ",len(fea_list))
    
    #count feature 
    for fea in fea_list:
        if fea in dic[i].keys():
            dic[i][fea] += 1
        else:
            dic[i][fea] = 1
    #statistic feature
    for keys, values in dic[i].items():
        total_sum += int(keys) * int(values)
        total_square_sum += pow(int(keys),2) * int(values)
        total_click += int(values)
        if int(keys) < minimum:
            minimum = int(keys)
        if int(keys) > maximum:
            maximum = int(keys)
    mean = total_sum/total_click
    variance = (total_square_sum/total_click) - pow(mean,2)
    ##record
    inform_list[i]['total_sum'] = total_sum
    inform_list[i]['total_square_sum'] = total_square_sum
    inform_list[i]['total_click'] = total_click
    inform_list[i]['unknown'] = unknown
    inform_list[i]['mean'] = mean
    inform_list[i]['variance'] = variance
    inform_list[i]['min'] = minimum
    inform_list[i]['max'] = maximum
    new_max = get_field_inform(total_data = inform_list, field_num = i, percentile = True)
    print('\033[91m' + 'Redundant Rate: ',((old_max - new_max) / old_max) * 100,"%" + '\033[0m')

if args.save is True:
    with open(args.save_path + "kaggle_dense_fea_count_" + f"percentile_{args.percentile}.pkl", "wb") as file:
        pickle.dump(dic,file)
    with open(args.save_path + "statistics_dense_fea_" + f"percentile_{args.percentile}.pkl", "wb") as file:
        pickle.dump(inform_list,file)
    print("***********")
    print("SAVE FINISH")
    print("***********")
