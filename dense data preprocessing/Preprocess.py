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

import random, math, os
import matplotlib.pyplot as plt
import math
import pickle
random.seed(0)

# +

# Original dataset
###############################################################################
ori_train_data = "/work/twsugkm569/data/kaggle/dataset_before_preprocess/train.txt"
split_train_data = "/work/twsugkm569/data/kaggle/dataset_before_preprocess/train_s.txt"
split_valid_data = "/work/twsugkm569/data/kaggle/dataset_before_preprocess/valid_s.txt"
###############################################################################

# log(float(x))**2 dataset (3 Idiots)
#train_data_csv = "/work/twsugkm569/data/kaggle/dataset_3idiots/train.csv"
#valid_data_csv = "/work/twsugkm569/data/kaggle/dataset_3idiots/valid.csv"

#Clamp
percentile = 9925
train_data_csv = f"/work/twsugkm569/data/kaggle/dataset_mirror_normalize{percentile}/train.csv"
valid_data_csv = f"/work/twsugkm569/data/kaggle/dataset_mirror_normalize{percentile}/valid.csv"
with open(f"/work/twsugkm569/data/kaggle/statistic/statistics_dense_fea_percentile_0.{percentile}.pkl","rb") as file:
    x = pickle.load(file)
#
#train_data_csv = "/work/twsugkm569/data/kaggle/dataset_normalize995/train.csv"
#valid_data_csv = "/work/twsugkm569/data/kaggle/dataset_normalize995/valid.csv"
# Normalized dataset
#train_data_csv = "/work/twsugkm569/data/kaggle_before_preprocess/train.csv"
#valid_data_csv = "/work/twsugkm569/data/kaggle_before_preprocess/valid.csv"

# Dataset without any preprocess
#train_data_csv = "/home/twsugkm569/Gary/org_DeepLight/data/train.csv"
#valid_data_csv = "/home/twsugkm569/Gary/org_DeepLight/data/valid.csv"

#train_data_csv = "/work/twsugkm569/data/kaggle_percentile_mean_0.99/train.csv"
#valid_data_csv = "/work/twsugkm569/data/kaggle_percentile_mean_0.99/valid.csv"
#train_data_csv = "/work/twsugkm569/data/kaggle_percentile_clamp_0.995_with_outlier/train.csv"
#valid_data_csv = "/work/twsugkm569/data/kaggle_percentile_clamp_0.995_with_outlier/valid.csv"
#train_data_csv = "/work/twsugkm569/data/kaggle_percentile_clamp_0.995_with_outlier/train.csv"
#valid_data_csv = "/work/twsugkm569/data/kaggle_percentile_clamp_0.995_with_outlier/valid.csv"

#train_data_csv = "/work/twsugkm569/data/kaggle_psuedoN_bias1_percentile_clamp_0.995_with_outlier/train.csv"
#valid_data_csv = "/work/twsugkm569/data/kaggle_psuedoN_bias1_percentile_clamp_0.995_with_outlier/valid.csv"

# +
# 3idiots log(float(x))**2 dataset
#def scale(x, i):
#    if x == '':
#        return '0'
#    elif float(x) > 2:
#        return str(int(math.log(float(x))**2))
#    else:
#        return x

# +
# Dataset without any preprocess

#def scale(x, i):
#    if x == '':
#        return '0'
#    else:
#        return x

# +
# Normalized dataset

#def scale(x, i):
#    mean = [3.5024133170754044
#            ,105.84841979766546
#            ,26.913041020611274
#            ,7.322680248873305
#            ,18538.991664871523
#            ,116.06185085211598
#            ,16.333130032135028
#            ,12.517042137556713
#            ,106.1098234380509
#            ,0.6175294977722137
#            ,2.7328343170173044
#            ,0.9910356287721244
#            ,8.217461161174054 ]
#    std = [9.429076218877869
#            ,391.4578184172999
#            ,397.97257749551636
#            ,8.793230590188488
#            ,69394.60106925869
#            ,382.5664439963476
#            ,66.04975449211732
#            ,16.688884385665453
#            ,220.28309147514412
#            ,0.6840505417423962
#            ,5.1990708255393745
#            ,5.597723612336433
#            ,16.21193233240115 ]
#    if x == '':
#        return '0'
#    else:
#        x = float(x) - mean[i - 1]
#        x = x / std[i - 1]
#        return str(x)

# +
# Normalized dataset
# Clamp 99%
# Then Normalize

#def scale(x, i):
#    maximum = [39, 2240, 222, 40, 348664, 1286, 217, 48, 988, 3, 26, 19, 48]
#    mean = [3.219885598553858, 97.78606489960639, 17.96430374346167, 7.226079100141581, 16295.034490220009, 105.25517900003565, 13.9166026775974, 12.412114000824657, 100.05336249457935, 0.6143717220954586, 2.6038000443752805, 0.7858875017237033, 7.897601242093751]
#    std = [6.305633733397206, 321.62141772397666, 33.435071380468536, 8.169319630014163, 47618.17730667609, 201.9336213815333, 31.829393792152032, 13.01746744947401, 166.18146241690854, 0.6694842256552023, 4.177773108829172, 2.5900396838090085, 9.52490823348167]
#    if x == '':
#        return '0'
#    else:
#        if float(x) <= maximum[i - 1]:
#            x = float(x) - mean[i - 1]
#            x = x / std[i - 1]
#        else:
#            x = 0
#        return str(x)

# +
# Normalized dataset
# Clamp 99%
# Then Normalize
# With Outlier
#def scale(x, i):
#    #maximum = [39, 2240, 222, 40, 348664, 1286, 217, 48, 988, 3, 26, 19, 48]
#    mean = [3.219885598553858, 97.78606489960639, 17.96430374346167, 7.226079100141581, 16295.034490220009, 105.25517900003565, 13.9166026775974, 12.412114000824657, 100.05336249457935, 0.6143717220954586, 2.6038000443752805, 0.7858875017237033, 7.897601242093751]
#    std = [6.305633733397206, 321.62141772397666, 33.435071380468536, 8.169319630014163, 47618.17730667609, 201.9336213815333, 31.829393792152032, 13.01746744947401, 166.18146241690854, 0.6694842256552023, 4.177773108829172, 2.5900396838090085, 9.52490823348167]
#    if x == '':
#        return '0'
#    else:
#        #if float(x) <= maximum[i - 1]:
#        x = float(x) - mean[i - 1]
#        x = x / std[i - 1]
#        #else:
#            #x = 0
#        return str(x)
# -

#with open("/work/twsugkm569/data/kaggle/statistic/statistics_dense_fea_percentile_0.990.pkl","rb") as file:
#    x = pickle.load(file)
mean = [x[i]['mean'] for i in range(13)]
std = [math.sqrt(x[i]['variance']) for i in range(13)]
def scale(x, i):
    if x == '':
        return '0'
    else:
        #if float(x) <= maximum[i - 1]:
        x = float(x) #- mean[i - 1]
        x = x / std[i - 1]
        #else:
            #x = 0
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


def generate_feature_map_and_train_csv(inputs, train_csv, file_feature_map, freq_dict, threshold=4):
    feature_map = []
    for i in range(40):
        feature_map.append({})
        if i > 13 :
            #------------------------------------------------------------------plot freq of each feature
            sorted_freq = sorted(freq_dict[i].items(), key=lambda x:x[1])
            sorted_freq = [x[1] for x in sorted_freq]
            #print(sorted_freq)
            index = [j for j in range(len(sorted_freq))]            
            for idx , j in enumerate(sorted_freq):
                if j >= threshold:
                    thres_index = idx
                    break   
            #sorted_freq = np.array(sorted_freq)
            if len(sorted_freq) > 1000:
                plot_index = []
                plot_freq = []
                count = 0.0
                length = len(sorted_freq)
                for idx in range(length):
                    if idx/float(length) >= (count/1000):
                        count += 1
                        plot_index.append(idx)
                        plot_freq.append(sorted_freq[idx])
            else:
                plot_index = index
                plot_freq = sorted_freq
            #plt.plot(plot_index,plot_freq)
            #plt.axvline(x=thres_index,color='r')
            #plt.suptitle("Sorted frequency of feature , Embedding_layer"+str(i-13-1))
            #plt.title("% of index frequency < "+str(threshold)+" = "+"{:4.1f}".format(thres_index/len(sorted_freq) *100)+"%")
            #plt.xlabel("index")
            #plt.ylabel("frequency")
            #plt.savefig('freq_cal/embedding_'+str(i-13-1)+'_freq_cal.png')
            #plt.clf()
            print("Embedding layer  " + "{:2d}".format(i-13-1) +" : total index = # = " + str(len(sorted_freq)))
            print("                      Deleted index # = " + str(thres_index))
            print("                      Compress rate % = " + str(thres_index/len(sorted_freq) *100))
            #------------------------------------------------------------------

    fout = open(train_csv, 'w')
    for idx, line in enumerate(open(inputs)):
        line = line.replace('\n', '').split('\t')
        if idx % 1000000 == 0 and idx > 0:
            print(idx)
        output_line = [line[0]]
        for i in range(1, 40):
            # map numerical features
            if i < 14:
                #line[i] = project_numeric(line[i])
                line[i] = scale(line[i], i)
                output_line.append(line[i])
            # handle categorical features
            elif freq_dict[i][line[i]] < threshold:
                output_line.append('0')
            elif line[i] in feature_map[i]:
                output_line.append(feature_map[i][line[i]])
            else:
                output_line.append(str(len(feature_map[i]) + 1))
                feature_map[i][line[i]] = str(len(feature_map[i]) + 1)
        output_line = ','.join(output_line)
        fout.write(output_line + '\n')

    # write feature_map file
    f_map = open(file_feature_map, 'w')
    for i in range(1, 40):
        #only_one_zero_index = True
        for feature in feature_map[i]:
            #if feature_map[i][feature] == '0' and only_one_zero_index == False:
            #    continue
            f_map.write(str(i) + ',' + feature + ',' + feature_map[i][feature] + '\n')
            #if only_one_zero_index == True and feature_map[i][feature] == '0':
            #    only_one_zero_index = False
    return feature_map


def generate_valid_csv(inputs, valid_csv, feature_map):
    fout = open(valid_csv, 'w')
    for idx, line in enumerate(open(inputs)):
        line = line.replace('\n', '').split('\t')
        output_line = [line[0]]
        for i in range(1, 40):
            if i < 14:
                #line[i] = project_numeric(line[i])
                line[i] = scale(line[i], i)
                output_line.append(line[i])
            elif line[i] in feature_map[i]:
                output_line.append(feature_map[i][line[i]])
            else:
                output_line.append('0')
        output_line = ','.join(output_line)
        fout.write(output_line + '\n')


print('Load the frequency~')
#with open("fea_count.pkl", "rb") as file:
#    freq_dict = pickle.load(file)
freq_dict = cnt_freq_train(ori_train_data)
print('Load the frequency finish!')

# +
print('Generate the feature map and impute the training dataset~')
feature_map = generate_feature_map_and_train_csv(split_train_data, train_data_csv, 'criteo_feature_map_wo_pre', freq_dict, threshold=8)
#feature_map = generate_feature_map_and_train_csv(split_train_data, train_data_csv, 'criteo_feature_map_IP8', freq_dict, threshold=8)
print('Generate the feature map and impute the training dataset finish!')

print('Generate the feature map and impute the valid dataset~')
generate_valid_csv(split_valid_data, valid_data_csv, feature_map)
print('Generate the feature map and impute the valid dataset finish!')

#print('Delete unnecessary files')
#os.system('rm train1.txt valid.txt')
# -


