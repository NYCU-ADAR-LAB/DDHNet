# -*- coding:utf-8 -*-


"""
Created on Dec 10, 2017
@author: jachin,Nie

Edited by Wei Deng on Jun.7, 2019

A pytorch implementation of deepfms including: FM, FFM, FwFM, DeepFM, DeepFFM, DeepFwFM

Reference:
[1] DeepFwFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.

"""
"""
Edited by Zhu Yuda  
Implement Facebook dlrm to apply structural pruning in DeepFwFM
Reference :
facebookresearch/dlrm
https://github.com/facebookresearch/dlrm
"""
import os,sys,random
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time
import sklearn.metrics as metrics

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torch.backends.cudnn
import math

from .pep_embedding import PEPEmbedding
from tensorboardX import SummaryWriter
"""
    Network structure
"""

class DLRM(torch.nn.Module):
    """
    :parameter
    -------------
    field_size: size of the feature fields
    feature_sizes: a field_size-dim array, sizes of the feature dictionary
    embedding_size: size of the feature embedding
    is_shallow_dropout: bool, shallow part(fm or ffm part) uses dropout or not?
    dropout_shallow: an array of the size of 2, example:[0.5,0.5], the first element is for the-first order part and the second element is for the second-order part
    h_depth: deep network's hidden layers' depth
    deep_layers: a h_depth-dim array, each element is the size of corresponding hidden layers. example:[32,32] h_depth = 2
    is_deep_dropout: bool, deep part uses dropout or not?
    dropout_deep: an array of dropout factors,example:[0.5,0.5,0.5] h_depth=2
    deep_layers_activation: relu or sigmoid etc
    n_epochs: epochs
    batch_size: batch_size
    learning_rate: learning_rate
    optimizer_type: optimizer_type, 'adam', 'rmsp', 'sgd', 'adag'
    is_batch_norm：bool,  use batch_norm or not ?
    verbose: verbose
    weight_decay: weight decay (L2 penalty)
    use_fm: bool
    use_ffm: bool
    use_deep: bool
    loss_type: "logloss", only
    eval_metric: roc_auc_score
    use_cuda: bool use gpu or cpu?
    n_class: number of classes. is bounded to 1
    greater_is_better: bool. Is the greater eval better?


    Attention: only support logsitcs regression
    """
    def __init__(self, field_size, feature_sizes, embedding_size = 64, is_shallow_dropout = True, dropout_shallow = [0.0,0.0],
                 h_depth = 3, is_deep_dropout = False, dropout_deep=[0.5, 0.5, 0.5, 0.5,0.5,0.5], eval_metric = roc_auc_score,
                 deep_layers_activation = 'relu', n_epochs = 64, batch_size = 2048, learning_rate = 0.001, momentum = 0.9,
                 optimizer_type = 'adam', is_batch_norm = True, verbose = False, random_seed = 0, weight_decay = 0.0,
                 loss_type = 'logloss',use_cuda = True, n_class = 1, greater_is_better = True, sparse = 0.9, warm = 10, 
                 numerical=13, top_mlp='1024-512-256' , bottom_mlp='512-256',field_weight = False,is_bias = True,
                 full_interactions = False, interaction_op = 'dot',
                 threshold_type = 'dimension', threshold_init = -150, emb_save_path='./tmp/embedding/{task}'):
        super(DLRM, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.is_shallow_dropout = is_shallow_dropout
        self.dropout_shallow = dropout_shallow
        self.is_deep_dropout = is_deep_dropout
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.optimizer_type = optimizer_type
        self.is_batch_norm = is_batch_norm
        self.verbose = verbose
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.use_cuda = use_cuda
        self.n_class = n_class
        self.greater_is_better = greater_is_better
        self.target_sparse = sparse
        self.warm = warm
        self.num = numerical
        self.print_weight = False
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        self.top_mlp = [int(i) for i in top_mlp.split('-')]
        self.bottom_mlp = [int(i) for i in bottom_mlp.split('-')]
        self.is_bias = is_bias
        self.interaction_op = interaction_op
        
        if self.interaction_op=='dot':
            if full_interactions :
                self.interaction_size = (self.field_size - self.num+1)**2
            else :
                self.interaction_size = int((self.field_size - self.num+1) * (self.field_size - self.num)/2)
        elif self.interaction_op=='cat' or self.interaction_op=='gate':
            self.interaction_size = self.embedding_size*26
        else:
            sys.exit(
                "ERROR: --interaction_op="
                + interaction_op
                + " is not supported"
            )
        self.top_mlp_insize = self.interaction_size + self.embedding_size
        self.field_weight = field_weight
        """
            check cuda
        """
        if self.use_cuda and not torch.cuda.is_available():
            self.use_cuda = False
            print("Cuda is not available, automatically changed into cpu model")

        """
            LR/fm/fwfm part
        """
        # self.dlrm_embeddings = nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
        self.candidate_p = [int(sum(self.feature_sizes)*self.embedding_size * factor) for factor in [0.5, 0.4, 0.3, 0.2, 0.125, 0.1, 0.05, 0.04, 0.03125, 0.03, 0.02, 0.01]]
        print('candidate_p: ', self.candidate_p)
        self.opt = {
            'use_cuda': self.use_cuda,
            'latent_dim': self.embedding_size,
            'threshold_type': threshold_type,
            'threshold_init': threshold_init,
            'emb_save_path': emb_save_path + '/sparsity_{overall_sparsity}_total_params_{num_parameter}_{emb}'
        }
        
        self._writer = SummaryWriter(log_dir=emb_save_path.replace('embedding', 'runs'))
        self.dlrm_embeddings = nn.ModuleList([PEPEmbedding(self.opt, emb, idx_num) for emb, idx_num in enumerate(self.feature_sizes)])
        self.emb_dropout = nn.Dropout(self.dropout_shallow[0])
        if self.field_weight:
            self.field_cov = nn.Linear((self.field_size - self.num+1),(self.field_size - self.num+1), bias=False)
            
        if self.interaction_op=='gate':
            setattr(self, 'emb_gate_batch_norm', nn.BatchNorm1d(self.top_mlp_insize, momentum=0.005)) #interaction_size
            setattr(self, 'emb_gate_linear', nn.Linear(self.top_mlp_insize, self.top_mlp_insize)) #interaction_size

        """
            deep parts
        """
        print("Init deep part")
        """
        bottom mlp part
        """
        #first layer
        setattr(self, 'net_' + 'bottom_linear_'+str(0), nn.Linear(self.num, self.bottom_mlp[0]))
        if self.is_batch_norm:
            setattr(self, 'net_' + 'bottom_batch_norm_' + str(0), nn.BatchNorm1d(self.bottom_mlp[0], momentum=0.005))
        if self.is_deep_dropout:
            setattr(self, 'net_' + 'bottom_linear_'+str(0)+'_dropout', nn.Dropout(self.dropout_deep[0]))
        #middle layer
        for i,o in enumerate(self.bottom_mlp[1:],1):
            setattr(self, 'net_' + 'bottom_linear_'+str(i), nn.Linear(self.bottom_mlp[i-1], self.bottom_mlp[i]))
            if self.is_batch_norm:
                setattr(self, 'net_' + 'bottom_batch_norm_' + str(i), nn.BatchNorm1d(self.bottom_mlp[i], momentum=0.005))
            if self.is_deep_dropout:
                setattr(self, 'net_' + 'bottom_linear_'+str(i)+'_dropout', nn.Dropout(self.dropout_deep[i]))
        #last layer 
        setattr(self, 'net_' + 'bottom_linear_'+'out', nn.Linear(self.bottom_mlp[-1], self.embedding_size))
        if self.is_batch_norm:
            setattr(self, 'net_' + 'bottom_batch_norm_' + 'out', nn.BatchNorm1d(self.embedding_size, momentum=0.005))
        if self.is_deep_dropout:
            setattr(self, 'net_' + 'bottom_linear_'+'out'+'_dropout', nn.Dropout(self.dropout_deep[0]))
        """
        top mlp part
        """
        #first layer
        setattr(self, 'net_' + 'top_linear_'+str(0), nn.Linear(self.top_mlp_insize, self.top_mlp[0]))
        if self.is_batch_norm:
            setattr(self, 'net_' + 'top_batch_norm_' + str(0), nn.BatchNorm1d(self.top_mlp[0], momentum=0.005))
        if self.is_deep_dropout:
            setattr(self, 'net_' + 'top_linear_'+str(0)+'_dropout', nn.Dropout(self.dropout_deep[0]))
            
        #middle layer
        for i,o in enumerate(self.top_mlp[1:],1):
            setattr(self, 'net_' + 'top_linear_'+str(i), nn.Linear(self.top_mlp[i-1], self.top_mlp[i]))
            if self.is_batch_norm:
                setattr(self, 'net_' + 'top_batch_norm_' + str(i), nn.BatchNorm1d(self.top_mlp[i], momentum=0.005))
            if self.is_deep_dropout:
                setattr(self, 'net_' + 'top_linear_'+str(i)+'_dropout', nn.Dropout(self.dropout_deep[i]))
        #last layer
        setattr(self, 'net_' + 'top_linear_'+'out', nn.Linear(self.top_mlp[-1], 1))
        if self.is_batch_norm:
            setattr(self, 'net_' + 'top_batch_norm_' + 'out', nn.BatchNorm1d(1, momentum=0.005))
        if self.is_deep_dropout:
            setattr(self, 'net_' + 'top_linear_'+'out'+'_dropout', nn.Dropout(self.dropout_deep[0]))   

    def near_ts_seg(self,ts, ls):
        SEG_SIZE = 500000
        sz = ts.size()
        ts_copy = ts.clone()
        ts_view_ls = torch.chunk(ts_copy.view(-1,1),math.ceil(ts_copy.numel()/SEG_SIZE),0)
        for ts_seg in ts_view_ls:
            idx = torch.min(torch.abs(torch.Tensor(ls).cuda().view(1,-1) - ts_seg), dim=1)[1]
            ts_seg.copy_(torch.gather(torch.Tensor(ls).cuda(),0,idx).view(-1,1))
        return torch.cat(ts_view_ls,0).view(sz)    
    def gen_list_134(self,max_norm_exp):
        min_norm_exp = max_norm_exp - 6
        it = min_norm_exp
        L = []
        L.append(0.)
        L.append(                                     + 2.**(it-4))
        L.append(                        + 2.**(it-3)             )
        L.append(                        + 2.**(it-3) + 2.**(it-4))
        L.append(           + 2.**(it-2)                          )
        L.append(           + 2.**(it-2) +            + 2.**(it-4))
        L.append(           + 2.**(it-2) + 2.**(it-3)             )
        L.append(           + 2.**(it-2) + 2.**(it-3) + 2.**(it-4))
        L.append(2.**(it-1)                                       )
        L.append(2.**(it-1)                           + 2.**(it-4))
        L.append(2.**(it-1)              + 2.**(it-3)             )
        L.append(2.**(it-1)              + 2.**(it-3) + 2.**(it-4))
        L.append(2.**(it-1) + 2.**(it-2)                          )
        L.append(2.**(it-1) + 2.**(it-2) +            + 2.**(it-4))
        L.append(2.**(it-1) + 2.**(it-2) + 2.**(it-3)             )
        L.append(2.**(it-1) + 2.**(it-2) + 2.**(it-3) + 2.**(it-4))
    
        for loop in range(7):
            L.append(2.**it                                                    )
            L.append(2.**it                                        + 2.**(it-4))
            L.append(2.**it                           + 2.**(it-3)             )
            L.append(2.**it                           + 2.**(it-3) + 2.**(it-4))
            L.append(2.**it              + 2.**(it-2)                          )
            L.append(2.**it              + 2.**(it-2) +            + 2.**(it-4))
            L.append(2.**it              + 2.**(it-2) + 2.**(it-3)             )
            L.append(2.**it              + 2.**(it-2) + 2.**(it-3) + 2.**(it-4))
            L.append(2.**it + 2.**(it-1)                                       )
            L.append(2.**it + 2.**(it-1)                           + 2.**(it-4))
            L.append(2.**it + 2.**(it-1)              + 2.**(it-3)             )
            L.append(2.**it + 2.**(it-1)              + 2.**(it-3) + 2.**(it-4))
            L.append(2.**it + 2.**(it-1) + 2.**(it-2)                          )
            L.append(2.**it + 2.**(it-1) + 2.**(it-2) +            + 2.**(it-4))
            L.append(2.**it + 2.**(it-1) + 2.**(it-2) + 2.**(it-3)             )
            L.append(2.**it + 2.**(it-1) + 2.**(it-2) + 2.**(it-3) + 2.**(it-4))
            it = it + 1
    
        I = L.copy()
        I.reverse()
        for x in range(len(I)):
            I[x] = -1. * I[x]
        I.pop()
        for y in L:
            I.append(y)
        return I    
    def gen_list_143(self,max_norm_exp):
        min_norm_exp = max_norm_exp - 14
        it = min_norm_exp
        L = []
        L.append(0.)
        L.append(                        + 2.**(it-3)             )
        L.append(           + 2.**(it-2)                          )
        L.append(           + 2.**(it-2) + 2.**(it-3)             )
        L.append(2.**(it-1)                                       )
        L.append(2.**(it-1)              + 2.**(it-3)             )
        L.append(2.**(it-1) + 2.**(it-2)                          )
        L.append(2.**(it-1) + 2.**(it-2) + 2.**(it-3)             )
        for loop in range(15):
            L.append(2.**it                                                    )
            L.append(2.**it                           + 2.**(it-3)             )
            L.append(2.**it              + 2.**(it-2)                          )
            L.append(2.**it              + 2.**(it-2) + 2.**(it-3)             )
            L.append(2.**it + 2.**(it-1)                                       )
            L.append(2.**it + 2.**(it-1)              + 2.**(it-3)             )
            L.append(2.**it + 2.**(it-1) + 2.**(it-2)                          )
            L.append(2.**it + 2.**(it-1) + 2.**(it-2) + 2.**(it-3)             )
            it = it + 1
    
        I = L.copy()
        I.reverse()
        for x in range(len(I)):
            I[x] = -1. * I[x]
        I.pop()
        for y in L:
            I.append(y)
        return I 
    def FFP8_quant(self):
        model_sd = self.state_dict()
        for name in model_sd:
            if 'top_linear' in name or 'bot_linear' in name or 'embeddings' in name :
                param = model_sd[name]
                layers_ffp8_143 = ['.2.','.3.','.11.','.15.','.20.','.23.','.25.']
                max_exp = math.floor(math.log(torch.max(abs(param)),2))
                #y = [x[1] for x in total_param_exp_count]
                if 'emb' in name and any( v in name for v in layers_ffp8_143 ) : #abs(max_exp - y.index(max(y))) < 4 :
                    ffp8_mat = self.gen_list_143(max_exp)  
                    #print(name +" : use FFP8( 1 , 4 , 3 , "+str(15-max_exp)+" )")
                else :
                    ffp8_mat = self.gen_list_134(max_exp)
                    #print(name +" : use FFP8( 1 , 3 , 4 , "+str(7-max_exp)+" )")                        
                model_sd[name] = self.near_ts_seg(param, ffp8_mat)                
                if self.print_weight:    
                    print("FFP8 matrix",ffp8_mat)
                    print("state_dict",model_sd[name])
        if self.print_weight:    
            print("######################################################")
        self.load_state_dict(model_sd)
        if self.print_weight:    
            print(self.state_dict())
        print("Quant to FFP8 and continue")
    def forward(self, Xi, Xv):
        """
        :param Xi_train: index input tensor, batch_size * embedding_size * 1
        :return: the last output
        """
       
        """
        DLRM part
        """
        """
        Bottom MLP part
        """
        if self.deep_layers_activation == 'sigmoid':
            activation = torch.sigmoid
        elif self.deep_layers_activation == 'tanh':
            activation = torch.tanh
        else:
            activation = torch.relu
        
        bottom_mlp_result = getattr(self, 'net_' + 'bottom_linear_'+str(0))(Xv)
        if self.is_batch_norm:
            bottom_mlp_result = getattr(self, 'net_' + 'bottom_batch_norm_' + str(0))(bottom_mlp_result)
        if self.is_deep_dropout:
            bottom_mlp_result = getattr(self, 'net_' + 'bottom_linear_'+str(0)+'_dropout')(bottom_mlp_result)
        bottom_mlp_result = activation(bottom_mlp_result)
        for i,o in enumerate(self.bottom_mlp[1:],1):
            bottom_mlp_result = getattr(self, 'net_' + 'bottom_linear_'+str(i))(bottom_mlp_result)
            if self.is_batch_norm:
                bottom_mlp_result = getattr(self, 'net_' + 'bottom_batch_norm_' + str(i))(bottom_mlp_result)
            if self.is_deep_dropout:
                bottom_mlp_result = getattr(self, 'net_' + 'bottom_linear_'+str(i)+'_dropout')(bottom_mlp_result)
            bottom_mlp_result = activation(bottom_mlp_result)
        bottom_mlp_result = getattr(self, 'net_' + 'bottom_linear_'+'out')(bottom_mlp_result)
        if self.is_batch_norm:
            bottom_mlp_result = getattr(self, 'net_' + 'bottom_batch_norm_' + 'out')(bottom_mlp_result)
        if self.is_deep_dropout:
            bottom_mlp_result = getattr(self, 'net_' + 'bottom_linear_'+'out'+'_dropout')(bottom_mlp_result)
        bottom_mlp_result = activation(bottom_mlp_result)
        
        #print('bottom_mlp_result',bottom_mlp_result.shape)
        """
        embedding
        """
        Tzero = torch.zeros(Xi.shape[0], 1, dtype=torch.long)
        if self.use_cuda:
            Tzero = Tzero.cuda()
        embedding_array = [torch.sum(emb(Xi[:,i,:]),1) for i, emb in enumerate(self.dlrm_embeddings)]    
        
        if self.interaction_op=='cat':
            dlrm_emb_vec = torch.cat(embedding_array,1)
            
        elif self.interaction_op=='gate':
            emb_vec = torch.cat(embedding_array,1)
            # print('emb_vec', emb_vec.shape)
            interaction_vec = torch.cat((emb_vec,bottom_mlp_result), 1)
            bn_result = getattr(self, 'emb_gate_batch_norm')(interaction_vec) #(emb_vec)
            fc_result = getattr(self, 'emb_gate_linear')(bn_result)
            act_result = torch.sigmoid(fc_result)
            dlrm_emb_vec = act_result * interaction_vec #emb_vec
        
        elif self.interaction_op=='dot':
            dlrm_emb_vec = torch.stack(embedding_array)

        """
        interaction layer
        """
        if self.interaction_op == 'dot':
            '''
            #27*2048*64
            interaction_vec = torch.cat((dlrm_emb_vec,bottom_mlp_result.unsqueeze(0))) 
            #print('interaction_vec',interaction_vec.shape)
            #27*2048*64  *   27*2048*64  ->  27*27*2048*64
            outer_result = torch.einsum('kij,lij->klij',interaction_vec,interaction_vec)
            #27*27   *    27*27*2048*64    ->   27*27*2048*64
            if self.field_weight:
                outer_result = torch.einsum('klij,kl->klij', outer_result, (self.field_cov.weight.t() + self.field_cov.weight) * 0.5)
            #27*27   *    27*27*2048*64    ->   27*27*2048
            interaction_result = torch.einsum('klij->kli', outer_result)        
            #print('interaction_result',interaction_result.shape)
            #print('bottom_mlp_result',bottom_mlp_result.shape)
            #27*27*2048  ->   729*2048
            if self.field_weight:
                interaction_result = torch.reshape(interaction_result,(interaction_result.shape[0]*interaction_result.shape[1] , -1))
            else : 
                interaction_result_list = [] 
                for i in range(interaction_result.shape[0]):
                    for j in range(i):
                        interaction_result_list.append(interaction_result[i][j].unsqueeze(0))
                interaction_result = torch.cat(interaction_result_list)
            #print('interaction_result',interaction_result.shape)
            # 729*2048  +   2048*64   ->   2048*793
            top_mlp_input = torch.cat((interaction_result.transpose(0,1),bottom_mlp_result),1)
            '''
            
            (batch_size, d) = embedding_array[0].shape # 26*2048*64
            interaction_vec = torch.cat((embedding_array + [bottom_mlp_result]), dim=1).view((batch_size, -1, d)) # 27*2048*64 -> 2048*27*64
            # perform a dot product
            Z = torch.bmm(interaction_vec, torch.transpose(interaction_vec, 1, 2)) # 2048*27*64 * 2048*64*27 -> 2048*27*27
            _, ni, nj = Z.shape
            offset = 0 # offset = 1 if full_interactions else 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            top_mlp_input = torch.cat(([Zflat] + [bottom_mlp_result]), dim=1) # R = torch.cat([x] + [Zflat], dim=1)
            
        elif self.interaction_op=='gate':
            top_mlp_input = dlrm_emb_vec
        elif self.interaction_op=='cat':
            top_mlp_input = torch.cat((dlrm_emb_vec,bottom_mlp_result), 1)
        """
        Top MLP part
        """
        top_mlp_result = getattr(self, 'net_' + 'top_linear_'+str(0))(top_mlp_input)
        if self.is_batch_norm:
            top_mlp_result = getattr(self, 'net_' + 'top_batch_norm_' + str(0))(top_mlp_result)
        if self.is_deep_dropout:
            top_mlp_result = getattr(self, 'net_' + 'top_linear_'+str(0)+'_dropout')(top_mlp_result)
        top_mlp_result = activation(top_mlp_result)    
        #middle layer
        for i,o in enumerate(self.top_mlp[1:],1):
            top_mlp_result = getattr(self, 'net_' + 'top_linear_'+str(i))(top_mlp_result)
            if self.is_batch_norm:
                top_mlp_result = getattr(self, 'net_' + 'top_batch_norm_' + str(i))(top_mlp_result)
            if self.is_deep_dropout:
                top_mlp_result = getattr(self, 'net_' + 'top_linear_'+str(i)+'_dropout')(top_mlp_result)
            top_mlp_result = activation(top_mlp_result)
        #last layer
        top_mlp_result = getattr(self, 'net_' + 'top_linear_'+'out')(top_mlp_result)
        if self.is_batch_norm:
            top_mlp_result = getattr(self, 'net_' + 'top_batch_norm_' + 'out')(top_mlp_result)
        if self.is_deep_dropout:
            top_mlp_result = getattr(self, 'net_' + 'top_linear_'+'out'+'_dropout')(top_mlp_result) 
        
        result = top_mlp_result.squeeze()
        return result


#
    # credit to https://github.com/ChenglongChen/tensorflow-DeepFM/blob/master/DeepFM.py
    def init_weights(self):
        model = self.train()
        require_update = True
        last_layer_size = 0
        TORCH = torch.cuda if self.use_cuda else torch
        for name, param in model.named_parameters():
            # if 'embeddings' in name:
                # param.data = TORCH.FloatTensor(param.data.size()).normal_().mul(0.01)
            if ('weight' in name) and ('embeddings' not in name): # weight and bias in the same layer share the same glorot
                glorot =  np.sqrt(2.0 / np.sum(param.data.shape))
                param.data = TORCH.FloatTensor(param.data.size()).normal_().mul(glorot)
            elif 'field_cov.weight' == name:
                param.data = TORCH.FloatTensor(param.data.size()).normal_().mul(np.sqrt(2.0 / (self.field_size - self.num+1) / 2))

    def calc_sparsity(self):
        base = sum(self.feature_sizes) * self.embedding_size
        non_zero_values = 0
        for i in range(26):
            non_zero_values += torch.nonzero(self.dlrm_embeddings[i].sparse_v, as_tuple=False).size(0)
        percentage = 1 - (non_zero_values / base)
        return percentage, non_zero_values
        
    def get_threshold(self):
        thr_list=[]
        for i in range(26):
            thr_list.append(torch.sigmoid(self.dlrm_embeddings[i].s))
        return thr_list
    
    def save_pruned_embedding(self, sparsity, param, step_idx):
        max_candidate_p = max(self.candidate_p)
        if max_candidate_p == 0:
            print("Minimal target parameters achieved, stop pruning.")
            exit(0)
        else:
            if param <= max_candidate_p:
                # embedding = self._factorizer.model.get_embedding()
                emb_save_path = self.opt['emb_save_path'].format(overall_sparsity=sparsity, num_parameter=param, emb='{emb}')
                emb_save_dir, _ = os.path.split(emb_save_path)
                if not os.path.exists(emb_save_dir):
                    os.makedirs(emb_save_dir)
                
                # np.save(emb_save_path, embedding)
                for i in range(26):
                    embedding = self.dlrm_embeddings[i].sparse_v.detach().cpu().numpy()
                    np.save(emb_save_path.format(emb=i), embedding)
                    
                max_idx = self.candidate_p.index(max(self.candidate_p))
                self.candidate_p[max_idx] = 0
                print("*" * 80)
                print("Reach the target parameter: {}, save embedding with size: {}".format(max_candidate_p, param))
                print("*" * 80)

    def fit(self, Xi_train, Xv_train, y_train, Xi_valid=None, Xv_valid=None, 
            y_valid = None, early_stopping=True, save_path = None, 
            FFP8 = True , quant_start_epoch = 10 , quant_freq = 100):
        """
        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                        indi_j is the feature index of feature field j of sample i in the training set
        :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                        vali_j is the feature value of feature field j of sample i in the training set
                        vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
        :param y_train: label of each sample in the training set
        :param Xi_valid: list of list of feature indices of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :param early_stopping: perform early stopping or not
        :param refit: refit the model on the train+valid dataset or not
        :param save_path: the path to save the model
        :return:
        """
        """
        pre_process
        """

        '''
        if save_path and not os.path.exists('/'.join(save_path.split('/')[0:-1])):
            print("Save path is not existed!")
            return
        '''
        Best_auc = 0
        Best_acc = 0
        Best_logloss = np.inf
        Best_model = False
        flag = 0
        if self.verbose:
            print("pre_process data ing...")
        is_valid = False
        Xi_train = np.array(Xi_train).reshape((-1, self.field_size-self.num, 1))
        Xv_train = np.array(Xv_train)
        y_train = np.array(y_train)
        x_size = Xi_train.shape[0]
        if Xi_valid:
            Xi_valid = np.array(Xi_valid).reshape((-1, self.field_size-self.num, 1))
            Xv_valid = np.array(Xv_valid)
            y_valid = np.array(y_valid)
            x_valid_size = Xi_valid.shape[0]
            is_valid = True
        if self.verbose:
            print("pre_process data finished")

        print('init_weights')
        self.init_weights()

        """
            train model
        """
        model = self.train()

        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'rmsp':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'adag':
            optimizer = torch.optim.Adagrad(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = F.binary_cross_entropy_with_logits

        train_result = []
        valid_result = []
        num_total = 0
        num_embeddings = 0
        num_dnn = 0
        print('========')
        for name, param in model.named_parameters():
            print (name, param.data.shape)
            num_total += np.prod(param.data.shape)
            if 'embeddings' in name:
                num_embeddings += np.prod(param.data.shape)
            if 'linear_' in name:
                num_dnn += np.prod(param.data.shape)
        print('Summation of feature sizes: %s' % (sum(self.feature_sizes)))
        print('Number of 1st order embeddings: %d' % (num_embeddings))
        #print('Number of 2nd order embeddings: %d' % (num_2nd_order_embeddings))
        print('Number of DNN parameters: %d' % (num_dnn))
        print("Number of total parameters: %d"% (num_total))
        n_iter = 0
        """
        Training + Pruning -non structual 
        """
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            batch_iter = x_size // self.batch_size
            epoch_begin_time = time()
            batch_begin_time = time()
            for i in range(batch_iter+1):
                if epoch >= self.warm:
                    n_iter += 1
                offset = i*self.batch_size
                end = min(x_size, offset+self.batch_size)
                if offset == end:
                    break
                batch_xi = Variable(torch.LongTensor(Xi_train[offset:end]))
                batch_xv = Variable(torch.FloatTensor(Xv_train[offset:end]))
                batch_y = Variable(torch.FloatTensor(y_train[offset:end]))
                if self.use_cuda:
                    batch_xi, batch_xv, batch_y = batch_xi.cuda(), batch_xv.cuda(), batch_y.cuda()
                optimizer.zero_grad()
                outputs = model(batch_xi, batch_xv)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.data.item()
                if self.verbose and i % 100 == 99:
                    eval = self.evaluate(batch_xi, batch_xv, batch_y)
                    print('[%d, %5d] loss: %.6f metric: %.6f time: %.1f s' %
                          (epoch + 1, i + 1, total_loss/100.0, eval, time()-batch_begin_time))
                    total_loss = 0.0
                    batch_begin_time = time()
                if FFP8 and epoch >= quant_start_epoch and (i == batch_iter or i % quant_freq == 9):
                    self.FFP8_quant()

                # Logging & Evaluate on the Evaluate Set
                step_idx = epoch*self.batch_size + i
                sparsity, params = self.calc_sparsity()  # 1 - (non_zero_values / base), non_zero_values
                self.save_pruned_embedding(sparsity, params, step_idx)
                self._writer.add_scalar('train/step_wise/mf_loss', loss.item(), step_idx)
                self._writer.add_scalar('train/step_wise/sparsity', sparsity, step_idx)
    
                if i % 2000 == 0:
                    print('[Epoch {}|Step {}|Flag {}|Sparsity {:.4f}|Params {}]'.format(epoch+1, step_idx+1, flag, sparsity, params))
          
            thr_list = self.get_threshold()
            for i in range(26):
                self._writer.add_histogram('threshold/'+str(i), thr_list[i], epoch)
            sparsity, params = self.calc_sparsity()
            self._writer.add_scalar('train/epoch_wise/sparsity', sparsity, epoch)
            self._writer.add_scalar('train/epoch_wise/params', params, epoch)
        
            # no_non_sparse = 0
            # for name, param in model.named_parameters():
                # no_non_sparse += (param != 0).sum().item()
            # print('Model parameters %d, sparse rate %.2f%%' % (no_non_sparse, 100 - no_non_sparse * 100. / num_total))
            train_loss, train_eval ,train_acc= self.eval_by_batch(Xi_train,Xv_train,y_train,x_size)
            train_result.append(train_eval)
            if is_valid:
                valid_loss, valid_eval ,valid_acc= self.eval_by_batch(Xi_valid, Xv_valid, y_valid, x_valid_size)
                valid_result.append(valid_eval)                
                if valid_eval > Best_auc :
                    Best_model = True
                    Best_auc = valid_eval
                    test_flag = 0
                else:
                    test_flag += 1
                    
                if valid_acc > Best_acc :
                    Best_model = True
                    Best_acc = valid_acc
                    
                if valid_loss < Best_logloss:
                    Best_model = True
                    Best_logloss = valid_loss
                    
                  
                print("*" * 80)
                print('Embedding Sparsity %.2f%%, Parameters %d' % (sparsity, params))
                print('Training [%d] Logloss: %.6f|AUC: %.6f|Acc: %.6f|Overall Sparsity %.2f%%|time: %.1f s' %
                  (epoch + 1, train_loss, train_eval,train_acc, 100 - (num_dnn+params) * 100. / num_total, time()-epoch_begin_time))
            
                print('Validation [%d] Logloss: %.6f|AUC: %.6f|Acc: %.6f|Overall Sparsity %.2f%%|time: %.1f s' %
                      (epoch + 1, valid_loss, valid_eval, valid_acc, 100 - (num_dnn+params) * 100. / num_total, time()-epoch_begin_time))
                if Best_model : 
                    print('Best AUC: %.6f ACC: %.6f LogLoss: %.6f' % (Best_auc , Best_acc, Best_logloss))
                print("*"*80)
            
            flag = test_flag
            # if self.early_stop is not None and flag >= self.early_stop:
                # print("Early stop training process")
                # print("Best performance on valid data: ", {'Best_auc': Best_auc, 'Best_acc': Best_acc, 'Best_logloss': Best_logloss})
                # self._writer.add_text('best_valid_result', str({'Best_auc': Best_auc, 'Best_acc': Best_acc, 'Best_logloss': Best_logloss}), 0)
                # exit()
            
            permute_idx = np.random.permutation(x_size)
            Xi_train = Xi_train[permute_idx]
            Xv_train = Xv_train[permute_idx]
            y_train = y_train[permute_idx]
            print('Training dataset shuffled.')
            
            if save_path and Best_model:
                print('Saving best model at ',save_path)
                Best_model = False
                torch.save(self.state_dict(),save_path)
            if is_valid and early_stopping and self.training_termination(valid_result):
                print("early stop at [%d] epoch!" % (epoch+1))
                print("Best performance on valid data: ", {'Best_auc': Best_auc, 'Best_acc': Best_acc, 'Best_logloss': Best_logloss})
                self._writer.add_text('best_valid_result', str({'Best_auc': Best_auc, 'Best_acc': Best_acc, 'Best_logloss': Best_logloss}), 0)
                break
        num_total = 0
        num_embeddings = 0
        num_dnn = 0
        print('========')
        for name, param in model.named_parameters():
            num_total += (param != 0).sum().item()
            if 'embeddings' in name:
                num_embeddings += (param != 0).sum().item()
            #if '2nd_embeddings' in name:
            #    num_2nd_order_embeddings += (param != 0).sum().item()
            if 'linear_' in name:
                num_dnn += (param != 0).sum().item()
            if 'field_cov.weight' == name:
                symm_sum = 0.5 * (param.data + param.data.t())
                non_zero_r = (symm_sum != 0).sum().item()
        if not self.field_weight:
            non_zero_r = 0 
        print('Number of pruned embeddings: %d' % (num_embeddings))
        print('Number of pruned 2nd order interactions: %d' % (non_zero_r))
        print('Number of pruned DNN parameters: %d' % (num_dnn))
        print("Number of pruned total parameters: %d"% (num_total))
        
    def post_eval(self,Xi_valid, Xv_valid, y_valid,load_path=None ):
        if load_path is not None:
            ld_model = torch.load(load_path)
            self.load_state_dict(ld_model)
            model = self.eval()
        else :
            return
        Xi_valid = np.array(Xi_valid).reshape((-1, self.field_size-self.num, 1))
        Xv_valid = np.array(Xv_valid)
        y_valid = np.array(y_valid)
        x_valid_size = Xi_valid.shape[0]
        #is_valid = True
        num_total = 0
        num_embeddings = 0
        #num_2nd_order_embeddings = 0
        num_dnn = 0
        print('========')
        for name, param in model.named_parameters():
            print (name, param.data.shape)
            num_total += np.prod(param.data.shape)
            if 'embeddings' in name:
                num_embeddings += np.prod(param.data.shape)
            if 'linear_' in name:
                num_dnn += np.prod(param.data.shape)
        print('Summation of feature sizes: %s' % (sum(self.feature_sizes)))
        print('Number of 1st order embeddings: %d' % (num_embeddings))
        #print('Number of 2nd order embeddings: %d' % (num_2nd_order_embeddings))
        print('Number of DNN parameters: %d' % (num_dnn))
        print("Number of total parameters: %d"% (num_total))
        print('*' * 50)
        no_non_sparse = 0
        for name, param in model.named_parameters():
            if 'embeddings' not in name:
                no_non_sparse += (param != 0).sum().item()
        sparsity, params = self.calc_sparsity()
        print('Model Parameters %d, Sparsity %.2f%%' % ((no_non_sparse+params), 100 - (no_non_sparse+params) * 100. / num_total))
        print('*' * 50)
        epoch_begin_time = time()
        valid_result = []
        valid_loss, valid_eval,valid_acc = self.eval_by_batch(Xi_valid, Xv_valid, y_valid, x_valid_size)
        valid_result.append(valid_eval)
        print('Validation Logloss: %.6f|AUC: %.6f|Acc: %.6f|Overall Sparsity %.2f%%|time: %.1f s' %
                  (valid_loss, valid_eval, valid_acc, 100 - (no_non_sparse+params) * 100. / num_total, time()-epoch_begin_time))
        print('*' * 50)
        
    def eval_by_batch(self,Xi, Xv, y, x_size):
        total_loss = 0.0
        y_pred = []

        batch_size = 8192
        batch_iter = x_size // batch_size
        criterion = F.binary_cross_entropy_with_logits
        model = self.eval()
        for i in range(batch_iter+1):
            offset = i * batch_size
            end = min(x_size, offset + batch_size)
            if offset == end:
                break
            batch_xi = Variable(torch.LongTensor(Xi[offset:end]))
            batch_xv = Variable(torch.FloatTensor(Xv[offset:end]))
            batch_y = Variable(torch.FloatTensor(y[offset:end]))
            if self.use_cuda:
                batch_xi, batch_xv, batch_y = batch_xi.cuda(), batch_xv.cuda(), batch_y.cuda()
            outputs = model(batch_xi, batch_xv)
            pred = torch.sigmoid(outputs).cpu()
            y_pred.extend(pred.data.numpy())
            loss = criterion(outputs, batch_y)
            total_loss += loss.data.item()*(end-offset)
        total_metric = self.eval_metric(y,y_pred)
        total_acc = metrics.accuracy_score(y,np.round(y_pred))
        return total_loss/x_size, total_metric , total_acc

    def binary_search_threshold(self, param, target_percent, total_no):
        l, r= 0., 1e2
        cnt = 0
        while l < r:
            cnt += 1
            mid = (l + r) / 2
            sparse_items = (abs(param) < mid).sum().item() * 1.0
            sparse_rate = sparse_items / total_no
            if abs(sparse_rate - target_percent) < 0.0001:
                return mid
            elif sparse_rate > target_percent:
                r = mid
            else:
                l = mid
            if cnt > 100:
                break
        return mid
    
    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def training_termination(self, valid_result):
        if len(valid_result) > 4:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                    valid_result[-2] < valid_result[-3] and \
                    valid_result[-3] < valid_result[-4]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                    valid_result[-2] > valid_result[-3] and \
                    valid_result[-3] > valid_result[-4]:
                    return True
        return False


    def predict(self, Xi, Xv):
        """
        :param Xi: the same as fit function
        :param Xv: the same as fit function
        :return: output, ont-dim array
        """
        Xi = np.array(Xi).reshape((-1,self.field_size - self.num,1))
        Xi = Variable(torch.LongTensor(Xi))
        Xv = Variable(torch.FloatTensor(Xv))
        if self.use_cuda and torch.cuda.is_available():
            Xi, Xv = Xi.cuda(), Xv.cuda()

        model = self.eval()
        pred = torch.sigmoid(model(Xi, Xv)).cpu()
        return (pred.data.numpy() > 0.5)

    def predict_proba(self, Xi, Xv):
        Xi = np.array(Xi).reshape((-1, self.field_size - self.num, 1))
        Xi = Variable(torch.LongTensor(Xi))
        Xv = Variable(torch.FloatTensor(Xv))
        if self.use_cuda and torch.cuda.is_available():
            Xi, Xv = Xi.cuda(), Xv.cuda()

        model = self.eval()
        pred = torch.sigmoid(model(Xi, Xv)).cpu()
        return pred.data.numpy()

    def inner_predict(self, Xi, Xv):
        """
        :param Xi: tensor of feature index
        :param Xv: tensor of feature value
        :return: output, numpy
        """
        model = self.eval()
        pred = torch.sigmoid(model(Xi, Xv)).cpu()
        return (pred.data.numpy() > 0.5)

    def inner_predict_proba(self, Xi, Xv):
        """
        :param Xi: tensor of feature index
        :param Xv: tensor of feature value
        :return: output, numpy
        """
        model = self.eval()
        pred = torch.sigmoid(model(Xi, Xv)).cpu()
        return pred.data.numpy()


    def evaluate(self, Xi, Xv, y):
        """
        :param Xi: tensor of feature index
        :param Xv: tensor of feature value
        :param y: tensor of labels
        :return: metric of the evaluation
        """
        y_pred = self.inner_predict_proba(Xi, Xv)
        return self.eval_metric(y.cpu().data.numpy(), y_pred)
