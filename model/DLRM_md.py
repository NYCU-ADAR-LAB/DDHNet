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
import matplotlib.pyplot as plt
# from tensorboardX import SummaryWriter

# +
def hard_gelu_v3(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    return (torch.heaviside(x + 2   , torch.cuda.FloatTensor([0.0])) - torch.heaviside(x + 0.25, torch.cuda.FloatTensor([0.0]))) * (x-0.25)*(x+2)/8 \
         + (torch.heaviside(x + 0.25, torch.cuda.FloatTensor([0.0])) - torch.heaviside(x - 2   , torch.cuda.FloatTensor([0.0]))) *  x*(x+2)/4 \
         + torch.heaviside(x - 2, torch.cuda.FloatTensor([0.0])) * x

class HardGELU_v3(nn.Module):
    def __init__(self, inplace=False):
        super(HardGELU_v3, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_gelu_v3(x, inplace=self.inplace)


# -

class CreateEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, base_dim, interaction_op):
        super(CreateEmbedding, self).__init__()
        self.interaction_op = interaction_op
        self.embs = nn.EmbeddingBag(
            num_embeddings, embedding_dim, mode="sum", sparse=False)
        torch.nn.init.xavier_uniform_(self.embs.weight)
        
        if self.interaction_op=='dot':
            if embedding_dim < base_dim:
                self.proj = nn.Linear(embedding_dim, base_dim, bias=False)
                torch.nn.init.xavier_uniform_(self.proj.weight)
            elif embedding_dim == base_dim:
                self.proj = nn.Identity()
            else:
                raise ValueError(
                    "Embedding dim " + str(embedding_dim) + " > base dim " + str(base_dim)
                )


    def forward(self, input):
        if self.interaction_op=='dot':
            return self.proj(self.embs(input, offsets=None))

        else:
            return self.embs(input, offsets=None)


init_step = 0  # for warmup learning rate
decay_init_step = 0 # for the rest decay part
learning_rate = 0
def adjust_learning_rate(optimizer, epoch=0, batch_size = 2048, init_lr = 0.001, n_epochs = 20, warmup = 0, alpha=0, schedule = 'cosine'):
        
    #Cosine learning rate decay
    ''' Per Batch warmup '''
    total_data = 41254892 # check your split train csv 2022/04/11
    step = total_data // batch_size + 1 # drop_liat == False
    global init_step, decay_init_step, learning_rate

    total_warmup_step = warmup*step

    if epoch < warmup and warmup != 0:
        init_step += 1
        lr = (init_lr * init_step) /warmup

    elif schedule == 'step':
        lr = init_lr * 0.3

    elif schedule == 'cosine':
        # per epoch cosine decay (HarDNet original)
        # lr = 0.5 * init_lr  * (1 + np.cos(np.pi * (epoch-warmup)/ (n_epochs-warmup)))

        # per batch cosine decay (me)



        total_decay_step = (n_epochs-warmup)*step

        # lr = 0.5 * args.lr  * (1 + np.cos(np.pi * (decay_init_step)/ (total_decay_step)))
        # decay_init_step += 1

        cosine_decay = 0.5  * (1 + np.cos(np.pi * (decay_init_step)/ (total_decay_step)))
        decayed = (1- alpha) * cosine_decay + alpha
        lr = init_lr * decayed
        decay_init_step += 1

    else: # no learning rate decay
        lr = init_lr
        
    # lr = 0.5 * args.lr  * (1 + np.cos(np.pi * (epoch)/ args.epochs ))

    learning_rate = lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
"""
    Network structure
"""

# +
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
    def __init__(self,field_size, feature_sizes, embedding_size = 64, is_shallow_dropout = True, dropout_shallow = [0.0,0.0],
                 h_depth = 3, is_deep_dropout = True, dropout_deep=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], eval_metric = roc_auc_score,
                 deep_layers_activation = 'relu', n_epochs = 64, batch_size = 2048, learning_rate = 0.001, momentum = 0.9,
                 optimizer_type = 'adam', is_batch_norm = False, verbose = False, random_seed = 0, weight_decay = 0.0,
                 loss_type = 'logloss',use_cuda = True, n_class = 1, greater_is_better = True, sparse = 0.9, warm = 10, 
                 numerical=13, top_mlp='1024-512-256' , bottom_mlp='512-256',field_weight = False, is_bias = True,
                 full_interactions = False, interaction_op = 'dot', md_sparsity='0.9', compressed_dim=8, expansion_factor=4, mix_act='relu'):
        super(DLRM, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        # PEP
        # md_emb_dict1={
        #     '0.9':    [20, 61,  3,  7, 33, 42, 47, 23, 64, 17, 51,  4, 62, 62, 33,  3, 64, 39, 44, 64,  3, 41, 60, 11, 38, 11],
        #     '0.96875':[ 4, 36,  0,  2,  7, 29, 20,  6, 47,  6, 19,  1, 40, 40, 14,  0, 50, 19, 22, 40,  1, 34, 45,  4, 23,  5],
        # }
        # DeepLight->PEP
        md_emb_dict2={
            '0.9':    [11, 32,  1,  8, 10, 28, 26, 10, 30, 18, 34,  7, 37, 39, 24,  5, 47, 27, 33, 30,  1, 29, 45, 14, 24, 17],
            '0.9375': [ 6, 27,  0,  4,  6, 24, 20,  5, 29, 12, 29,  4, 32, 36, 18,  3, 44, 20, 26, 28,  0, 25, 42,  9, 20, 12],
            '0.95':   [ 4, 24,  0,  3,  5, 23, 17,  4, 26, 10, 26,  3, 29, 33, 15,  2, 43, 17, 23, 26,  0, 24, 40,  7, 19, 10],
            '0.96875':[ 2, 18,  0,  2,  3, 21, 12,  3, 23,  7, 21,  2, 24, 29, 11,  1, 41, 12, 18, 23,  0, 21, 37,  5, 15,  6]
        }
        md16_emb_dict={
            '0.5':    [15, 16,  5, 10, 15, 14, 15, 16, 12, 12, 15,  9, 16, 15, 15,  5, 15, 15, 15, 13,  7, 15, 15, 12, 13, 12],
            '0.75':   [10, 16,  2,  6, 12, 12, 14, 12, 12,  8, 15,  4, 16, 15, 13,  2, 15, 13, 14, 13,  3, 13, 15,  8, 12,  8],
            '0.8':    [ 7, 16,  1,  4, 10, 11, 14,  9, 12,  8, 14,  3, 16, 15, 12,  1, 15, 12, 14, 13,  2, 13, 15,  7, 11,  7],
            '0.875':  [ 4, 15,  1,  3,  7, 11, 12,  5, 12,  6, 13,  2, 16, 15, 10,  1, 14, 11, 13, 13,  1, 11, 14,  5, 10,  5],
            '0.9':    [ 3, 14,  0,  2,  5, 10, 10,  4, 12,  5, 12,  1, 14, 14,  9,  0, 14,  9, 11, 12,  1, 10, 14,  4,  9,  4],
            '0.9375': [ 1,  9,  0,  2,  2,  8,  6,  1, 10,  3,  8,  1, 10, 12,  6,  0, 11,  6,  8,  9,  0,  8, 12,  3,  6,  3],
            '0.95':   [ 1,  8,  0,  1,  1,  7,  5,  1,  9,  2,  7,  1,  9, 10,  5,  0, 11,  4,  7,  7,  0,  8, 12,  2,  6,  2],
            '0.96875':[ 0,  5,  0,  1,  1,  6,  3,  0,  8,  1,  5,  0,  6,  9,  3,  0, 10,  3,  4,  6,  0,  7, 11,  2,  4,  1]
        }
        # Gate+PEP
        # md16_emb_dict2={
        #     '0.8':    [ 9, 16,  2,  4, 10, 11, 14, 10, 12,  7, 14,  2, 16, 15, 12,  2, 15, 12, 13, 13,  2, 12, 14,  6, 11,  7],
        #     '0.875':  [ 6, 15,  1,  3,  8, 11, 12,  7, 12,  6, 13,  1, 16, 15, 10,  1, 15, 10, 12, 13,  1, 10, 14,  4, 10,  5],
        # }
        # Index_Dimension PEP
        md16_emb_dict2={
            '0.8': [ 7, 16,  2,  4, 10, 11, 14, 10, 12,  7, 14,  3, 16, 15, 12,  1, 15, 13, 14, 13,  2, 13, 14,  7, 11,  7]
        }
        # FB->PEP
        md64_emb_dict={
            '0.8':    [32, 61,  9, 16, 38, 48, 49, 36, 64, 25, 54,  9, 62, 62, 38,  9, 64, 44, 50, 64, 11, 51, 61, 19, 44, 21],
            '0.875':  [22, 61,  4, 11, 27, 46, 43, 26, 64, 19, 51,  5, 61, 62, 33,  4, 64, 40, 46, 64,  6, 46, 61, 14, 40, 16],
            '0.9':    [19, 60,  3,  9, 24, 42, 41, 21, 64, 16, 49,  3, 61, 62, 31,  3, 64, 37, 44, 64,  4, 43, 61, 11, 37, 13],
            '0.95':   [11, 57,  1,  4, 18, 39, 31, 14, 64, 10, 43,  1, 57, 61, 22,  1, 64, 29, 35, 64,  2, 35, 60,  6, 33,  7],
            '0.96':   [ 9, 56,  1,  3, 15, 38, 27, 12, 64,  8, 40,  1, 55, 61, 20,  1, 63, 26, 32, 64,  1, 35, 60,  4, 32,  5],
            '0.96875':[ 7, 53,  0,  2, 13, 38, 21, 10, 64,  6, 37,  1, 53, 59, 17,  1, 58, 23, 28, 64,  1, 35, 60,  3, 31,  4],
            '0.97':   [ 7, 53,  0,  2, 12, 38, 20,  9, 64,  6, 36,  1, 52, 59, 16,  0, 58, 23, 27, 64,  1, 34, 60,  3, 31,  4],
            '0.975':  [ 6, 51,  0,  2, 11, 36, 17,  8, 64,  5, 33,  0, 49, 58, 14,  0, 58, 20, 24, 64,  1, 34, 59,  3, 30,  3],
            '0.98':   [ 5, 44,  0,  1,  9, 33, 12,  6, 60,  4, 27,  0, 43, 54, 11,  0, 57, 17, 20, 60,  1, 32, 58,  2, 28,  3],
            '0.985':  [ 3, 32,  0,  1,  6, 29,  8,  4, 47,  3, 20,  0, 35, 45,  8,  0, 50, 13, 16, 49,  0, 27, 49,  2, 22,  2],
            '0.99':   [ 2, 19,  0,  1,  4, 21,  5,  2, 31,  2, 12,  0, 24, 31,  5,  0, 40,  8, 11, 33,  0, 19, 35,  1, 14,  2],
            '0.995':  [ 1,  9,  0,  0,  2, 13,  2,  1, 21,  1,  6,  0, 11, 18,  3,  0, 31,  4,  5, 20,  0, 12, 23,  1,  7,  1],
        }
        md32_emb_dict={
            '0.8':    [16, 31,  5,  8, 18, 24, 25, 18, 32, 14, 27,  4, 31, 31, 20,  4, 32, 23, 25, 32,  4, 25, 32, 11, 22, 11],
            '0.875':  [11, 30,  3,  5, 14, 22, 23, 13, 32, 11, 25,  2, 31, 31, 17,  2, 32, 20, 23, 32,  2, 23, 31,  8, 20,  8],
            '0.9':    [ 9, 30,  2,  4, 13, 21, 21, 11, 32,  9, 24,  2, 30, 31, 16,  1, 32, 19, 22, 32,  2, 23, 30,  6, 19,  7],
            '0.95':   [ 6, 29,  1,  2,  9, 20, 16,  7, 32,  6, 21,  1, 29, 31, 11,  1, 32, 15, 18, 32,  1, 18, 30,  3, 17,  3],
            '0.96':   [ 4, 28,  0,  1,  7, 20, 13,  6, 32,  5, 19,  0, 28, 30, 10,  0, 29, 13, 16, 32,  0, 17, 30,  3, 16,  3],
            '0.96875':[ 4, 27,  0,  1,  6, 18, 10,  4, 32,  4, 16,  0, 26, 30,  8,  0, 29, 11, 14, 32,  0, 17, 30,  2, 15,  2],
            '0.97':   [ 3, 26,  0,  1,  6, 18, 10,  4, 31,  4, 15,  0, 25, 29,  8,  0, 29, 11, 13, 32,  0, 17, 29,  2, 15,  2],
            '0.975':  [ 3, 23,  0,  1,  5, 17,  8,  3, 28,  3, 12,  0, 22, 26,  7,  0, 27,  9, 11, 28,  0, 16, 27,  2, 13,  2],
            '0.98':   [ 2, 18,  0,  1,  3, 14,  6,  2, 25,  3,  8,  0, 19, 22,  6,  0, 24,  7,  9, 21,  0, 13, 24,  1, 11,  1],
            '0.985':  [ 1, 13,  0,  1,  2, 11,  4,  1, 22,  2,  6,  0, 14, 17,  4,  0, 20,  5,  7, 15,  0, 11, 19,  1,  8,  1],
            '0.99':   [ 1,  8,  0,  0,  1,  9,  2,  1, 16,  1,  3,  0, 10, 14,  3,  0, 18,  3,  5, 13,  0,  8, 15,  1,  6,  1],
            '0.995':  [ 0,  5,  0,  0,  0,  7,  1,  0, 14,  1,  1,  0,  6, 10,  2,  0, 15,  2,  2, 11,  0,  6, 13,  0,  4,  0],
        }
        if md_sparsity == '0':
            self.md_emb_sizes = [embedding_size]*26
        else:
            self.md_emb_sizes = md64_emb_dict[md_sparsity]
        self.is_shallow_dropout = is_shallow_dropout
        self.dropout_shallow = dropout_shallow
        self.is_deep_dropout = is_deep_dropout
        self.dropout_deep = dropout_deep
        if interaction_op=='cat':
            self.dropout_deep[0]=0.1
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
        self.compressed_dim = compressed_dim
        self.num_fea = self.field_size - self.md_emb_sizes.count(0) - self.num + 1

        if self.interaction_op=='dot' or self.interaction_op=='paddot':
            if full_interactions :
                self.interaction_size = (self.num_fea)**2
            else :
                self.interaction_size = int(self.num_fea * (self.num_fea - 1) / 2)
        elif self.interaction_op=='cat' or self.interaction_op=='gate':
            self.interaction_size = sum(self.md_emb_sizes)
        elif self.interaction_op=='mix':
            self.interaction_size = self.num_fea * self.compressed_dim
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
        self.dlrm_embeddings = nn.ModuleList()
        self.feat_zero_idx=[]
        for i in range(26):
            n = self.feature_sizes[i]
            m = self.md_emb_sizes[i]
            if m==0:
                self.feat_zero_idx.append(i)
                continue
            base = self.embedding_size
            EE = CreateEmbedding(n, self.md_emb_sizes[i], base, self.interaction_op)
            # use np initialization as below for consistency...
            W = np.random.uniform(
                low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
            ).astype(np.float32)
            EE.embs.weight.data = torch.tensor(W, requires_grad=True)
            self.dlrm_embeddings.append(EE)
            
        self.emb_dropout = nn.Dropout(self.dropout_shallow[0])
        if self.field_weight:
            self.field_cov = nn.Linear((self.field_size - self.num+1),(self.field_size - self.num+1), bias=False)
        
        if self.interaction_op=='gate':
            setattr(self, 'emb_gate_batch_norm', nn.BatchNorm1d(self.top_mlp_insize, momentum=0.005)) #interaction_size
            setattr(self, 'emb_gate_linear', nn.Linear(self.top_mlp_insize, self.top_mlp_insize)) #interaction_size
        elif self.interaction_op=='mix':
            # self.catego_mixing = nn.Linear(self.num_fea, self.num_fea)
            # self.latent_mixing = nn.Linear(self.embedding_size, self.compressed_dim)
            self.catego_mixing = nn.Sequential(
                nn.Linear(self.num_fea, self.num_fea * expansion_factor),
                # nn.GELU() if mix_act == "gelu" else nn.ReLU(),
                nn.Linear(self.num_fea * expansion_factor, self.num_fea),
            )
            self.latent_mixing = nn.Sequential(
                nn.Linear(self.embedding_size, self.embedding_size),
                # nn.GELU() if mix_act == "gelu" else nn.ReLU(),
                nn.Linear(self.embedding_size, self.compressed_dim),
            )
            
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
            if 'top_linear' in name or 'bottom_linear' in name or 'embeddings' in name :
                param = model_sd[name]
                # dlrm_48md_09
                layers_ffp8_143 = ['.0.embs','.4.embs','.5.embs','.7.embs','.9.embs','.15.embs','.20.embs','.21.embs','top_linear_0','top_linear_1','top_linear_2','bottom_linear']
                # dlrm_48md_cat_09_dropout01
                # layers_ffp8_143 = ['.3.embs','.8.embs','.11.embs','.19.embs','.21.embs','.24.embs','top_linear_0','top_linear_1','top_linear_2','bottom_linear']
                # dlrm_48md_096875
                # layers_ffp8_143 = ['.4.embs','.7.embs','.18.embs','.22.embs','top_linear_0','top_linear_1','top_linear_2','bottom_linear']

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
        if self.deep_layers_activation == 'hardgelu':
            activation = HardGELU_v3()
        elif self.deep_layers_activation == 'gelu':
            activation = torch.nn.functional.gelu
        elif self.deep_layers_activation == 'sigmoid':
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
        
        # print('bottom_mlp_result',bottom_mlp_result.shape)
        """
        embedding
        """
        Tzero = torch.zeros(Xi.shape[0], 1, dtype=torch.long)
        if self.use_cuda:
            Tzero = Tzero.cuda()
        # embedding_array = [torch.sum(emb(Xi[:,i,:]),1) for i, emb in enumerate(self.dlrm_embeddings)]
        # embedding_array = [emb(Xi[:,i,:]) for i, emb in enumerate(self.dlrm_embeddings)]
        k=0
        embedding_array=[]
        for j in range(26):
            if j in self.feat_zero_idx:
                continue
            V = self.dlrm_embeddings[k](Xi[:,j,:])
            if self.interaction_op=='mix' or self.interaction_op=='paddot':
                V = F.pad(V, (0, self.embedding_size - V.size(1)))
            embedding_array.append(V)
            k+=1
        #print(len(embedding_array))
        #print(embedding_array)
        #dlrm_emb_arr = torch.as_tensor(embedding_array)
        
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
        
        elif self.interaction_op=='dot' or self.interaction_op=='paddot':
            dlrm_emb_vec = torch.stack(embedding_array)
        elif self.interaction_op=='mix':
            # dlrm_emb_vec = torch.stack((embedding_array + [bottom_mlp_result]), 1) # shape=(b, 27, 16) for dim_mixing
            dlrm_emb_vec = torch.stack((embedding_array + [bottom_mlp_result]), 2) # shape=(b, 16, 27) for field_mixing
        # elif self.is_shallow_dropout: #True, 但是 Dropout(0)
        #     dlrm_emb_arr = torch.stack(embedding_array)
        #     #print(dlrm_emb_arr.shape)
        #     #26*2048*64
        #     dlrm_emb_vec = self.emb_dropout(dlrm_emb_arr) 
        
        
        """
        interaction layer
        """
        if self.interaction_op=='dot' or self.interaction_op=='paddot':
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
        elif self.interaction_op=='mix':
            out1 = self.catego_mixing(dlrm_emb_vec)
            out2 = self.latent_mixing(out1.transpose(1,2))
            dlrm_emb_vec = torch.flatten(out2, 1) # shape=(b, 27*8)
            top_mlp_input = torch.cat((dlrm_emb_vec, bottom_mlp_result), 1) # shape=(b, 27*8 + 16)

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
            """
            else:
                if (self.use_fwfm or self.use_fm) and require_update:
                    last_layer_size += (self.field_size + self.embedding_size)
                if self.use_deep and require_update:
                    last_layer_size += (self.deep_layers[-1] + 1)
                require_update = False
                if name in ['fm_1st.weight', 'fm_2nd.weight'] or 'fc.weight' in name:
                    param.data = TORCH.FloatTensor(param.data.size()).normal_().mul(np.sqrt(2.0 / last_layer_size))
            """

    def fit(self, Xi_train, Xv_train, y_train, Xi_valid=None, Xv_valid=None, 
            y_valid = None, ealry_stopping=False, refit=False, save_path = None, 
            prune=0, prune_emb=0, prune_r=0, prune_deep=0, prune_top=0, prune_bot=0, emb_r=1., emb_corr=1.,
           FFP8 = True , quant_start_epoch = 10 , quant_freq = 100, pretrained = None):
        """
        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                        indi_j is the feature index of feature field j of sample i in the training set
        :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                        vali_j is the feature value of feature field j of sample i in the training set
                        vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
        :param y_train: label of each sample in the training set
        :param Xi_valid: list of list of feature indices of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :param ealry_stopping: perform early stopping or not
        :param refit: refit the model on the train+valid dataset or not
        :param save_path: the path to save the model
        :param prune: control module to decide if to prune or not
        :param prune_fm: if prune the FM component
        :param prune_deep: if prune the DEEP component
        :param emb_r: ratio of sparse rate in FM over sparse rate in Deep
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
        Best_FFP8_auc = 0
        Best_FFP8_model = False
        Best_model = False
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

        if pretrained is not None:
            print('load pretrained weights', pretrained)
            self.load_state_dict(torch.load(pretrained))
        else:
            print('init_weights')
            self.init_weights()

        """
            train model
        """
        model = self.train()
        if self.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'adam':
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
        pruned_auc = [] 
        pruned_emb_rate = []
        #training_auc_record = []
        #testing_auc_record = []
        emb_current_pruned_rate = 0
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
        n_iter = 0
        """
        Training + Pruning -non structual 
        """
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            batch_iter = x_size // self.batch_size
            if FFP8 and epoch == quant_start_epoch:                    
                self.FFP8_quant()
                valid_loss, valid_eval, TPR, TNR, valid_acc, inf_time= self.eval_by_batch(Xi_valid, Xv_valid, y_valid, x_valid_size)                                
                print('Model After FFP8 Validation [%d] loss: %.6f AUC: %.6f TPR: %.6f TNR: %.6f Acc: %.6f sparse %.2f%% us/sample: %.3f' %
                      (epoch + 1, valid_loss, valid_eval, TPR, TNR, valid_acc, 100 - no_non_sparse * 100. / num_total, inf_time))
                optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate*0.7, weight_decay=self.weight_decay)
                # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate*0.5, weight_decay=self.weight_decay)
            if epoch == self.warm and self.warm != 0:
                # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate*0.8, weight_decay=self.weight_decay)
                optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate*0.3, weight_decay=self.weight_decay)
                # pt_ep20 = save_path.replace('prune_deep_75', 'ep20')
                # print('Saving model without pruning at', pt_ep20)
                # Best_model = False
                # torch.save(self.state_dict(), pt_ep20)
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
                # adjust_learning_rate(optimizer, batch_size=self.batch_size, init_lr=self.learning_rate, \
                #             alpha=self.lr_alpha, n_epochs=self.n_epochs, schedule=self.lr_decay)
                optimizer.zero_grad()
                outputs = model(batch_xi, batch_xv)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.data.item()
                if self.verbose and i % 1000 == 999:
                    eval = self.evaluate(batch_xi, batch_xv, batch_y)
                    print('[%d, %5d] loss: %.6f metric: %.6f time: %.1f s' %
                          (epoch + 1, i + 1, total_loss/100.0, eval, time()-batch_begin_time))
                    total_loss = 0.0
                    batch_begin_time = time()
                if FFP8 and epoch >= quant_start_epoch and (i == batch_iter or i % quant_freq == 9):
                    """  # Don't Print
                    if i == batch_iter:
                        self.print_weight = True
                    else :
                        self.print_weight = False
                    """
                    self.FFP8_quant()
                if prune and (i == batch_iter or i % 10 == 9) and epoch >= self.warm:
                    self.adaptive_sparse = self.target_sparse * (1 - 0.99**(n_iter /100.))
                    if prune_emb != 0:
                        stacked_embeddings = []
                        for name, param in model.named_parameters():
                            if 'embeddings' in name:
                                stacked_embeddings.append(param.data)
                        stacked_emb = torch.cat(stacked_embeddings, 0)
                        emb_threshold = self.binary_search_threshold(stacked_emb.data, self.adaptive_sparse * emb_r, np.prod(stacked_emb.data.shape))
                    for name, param in model.named_parameters():
                        if 'embeddings' in name and prune_emb != 0:
                            mask = abs(param.data) < emb_threshold
                            param.data[mask] = 0
                        
                        if 'field_cov.weight' == name and prune_r != 0:
                            layer_pars = np.prod(param.data.shape)
                            symm_sum = 0.5 * (param.data + param.data.t())
                            threshold = self.binary_search_threshold(symm_sum, self.adaptive_sparse * emb_corr, layer_pars)
                            mask = abs(symm_sum) < threshold
                            param.data[mask] = 0
                            #print (mask.sum().item(), layer_pars)
                            
                        if 'linear_' in name and 'weight' in name and prune_deep != 0:
                            layer_pars = np.prod(param.data.shape)
                            threshold = self.binary_search_threshold(param.data, self.adaptive_sparse, layer_pars)
                            mask = abs(param.data) < threshold
                            param.data[mask] = 0 
                            # if self.adaptive_sparse * emb_r > emb_current_pruned_rate : 
                                # print ('Pruned # params of',name,mask.sum().item(), layer_pars,'; Rate : %.3f' % (mask.sum().item()/layer_pars))
                        else:
                            if 'top_linear' in name and 'weight' in name and prune_top != 0:
                                layer_pars = np.prod(param.data.shape)
                                threshold = self.binary_search_threshold(param.data, self.adaptive_sparse, layer_pars)
                                mask = abs(param.data) < threshold
                                param.data[mask] = 0 
                                # if self.adaptive_sparse * emb_r > emb_current_pruned_rate : 
                                    # print ('Pruned # params of',name,mask.sum().item(), layer_pars,'; Rate : %.3f' % (mask.sum().item()/layer_pars))
                            if 'bottom_linear' in name and 'weight' in name and prune_bot != 0:
                                layer_pars = np.prod(param.data.shape)
                                threshold = self.binary_search_threshold(param.data, self.adaptive_sparse, layer_pars)
                                mask = abs(param.data) < threshold
                                param.data[mask] = 0
                                # if self.adaptive_sparse * emb_r > emb_current_pruned_rate : 
                                    # print ('Pruned # params of',name,mask.sum().item(), layer_pars,'; Rate : %.3f' % (mask.sum().item()/layer_pars))
                    ############## Record pruning rate versus auc
                    if self.adaptive_sparse * emb_r > emb_current_pruned_rate : 
                        pruned_emb_rate.append(emb_current_pruned_rate)                        
                        valid_loss, valid_eval, _, _, _, _= self.eval_by_batch(Xi_valid, Xv_valid, y_valid, x_valid_size)
                        pruned_auc.append(valid_eval)
                        print('Current pruning rate : %.6f ; Testing AUC = %.6f' %(self.adaptive_sparse * emb_r, valid_eval))
                        emb_current_pruned_rate += 0.005
                        
            epoch_end_time = time()
            no_non_sparse = 0
            for name, param in model.named_parameters():
                no_non_sparse += (param != 0).sum().item()
            print('Model parameters %d, sparse rate %.2f%%' % (no_non_sparse, 100 - no_non_sparse * 100. / num_total))
            train_loss, train_eval, _, _, train_acc, _= self.eval_by_batch(Xi_train,Xv_train,y_train,x_size)
            train_result.append(train_eval)
            print('Training [%d] loss: %.6f AUC: %.6f Acc: %.6f sparse %.2f%% time: %.3f s' %
                  (epoch + 1, train_loss, train_eval, train_acc, 100 - no_non_sparse * 100. / num_total, epoch_end_time-epoch_begin_time))
            if is_valid:
                valid_loss, valid_eval, TPR, TNR, valid_acc, inf_time= self.eval_by_batch(Xi_valid, Xv_valid, y_valid, x_valid_size)
                valid_result.append(valid_eval)                
                print('Validation [%d] loss: %.6f AUC: %.6f TPR: %.6f TNR: %.6f Acc: %.6f sparse %.2f%% us/sample: %.3f' %
                      (epoch + 1, valid_loss, valid_eval, TPR, TNR, valid_acc, 100 - no_non_sparse * 100. / num_total, inf_time))
                if valid_eval > Best_auc :
                    Best_model = True
                    Best_auc = valid_eval
                if valid_acc > Best_acc :
                    Best_acc = valid_acc
                    print('Best ACC: %.6f' % (Best_acc))
                if valid_eval > Best_FFP8_auc and  FFP8 and epoch >= quant_start_epoch:
                    Best_FFP8_auc = valid_eval
                    Best_FFP8_model = True 
                if Best_model:
                    print('Best AUC: %.6f ACC: %.6f' % (Best_auc , valid_acc))
            print('*' * 50)
            
            permute_idx = np.random.permutation(x_size)
            Xi_train = Xi_train[permute_idx]
            Xv_train = Xv_train[permute_idx]
            y_train = y_train[permute_idx]
            print('Training dataset shuffled.')
            
            if save_path and Best_model and (FFP8 and epoch >= quant_start_epoch)==False:
                print('Saving best model at ',save_path) # prune_deep.pt
                Best_model = False
                torch.save(self.state_dict(),save_path)
            if save_path and Best_FFP8_model:
                print('Saving best FFP8 model at ', save_path[:-3]+'_FFP8.pt')
                Best_FFP8_model = False
                torch.save(self.state_dict(), save_path[:-3]+'_FFP8.pt')
            if is_valid and ealry_stopping and self.training_termination(valid_result):
                print("early stop at [%d] epoch!" % (epoch+1))
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
        if prune:
            epochs = [i for i in range(self.n_epochs)]
            plt.plot(epochs,train_result,color='r')
            plt.plot(epochs,valid_result,color='g')
            plt.xticks(epochs)
            plt.title('Training AUC curve')
            plt.xlabel('epoch')
            plt.ylabel('AUC')
            plt.savefig('Training_AUC'+str(max(valid_result))+'.png')
            plt.clf()
            print(pruned_auc)
            plt.plot(pruned_emb_rate,pruned_auc)            
            plt.xticks(pruned_emb_rate)
            plt.title('Pruning rate versus AUC curve')
            plt.xlabel('Pruning rate(embedding layer)')
            plt.ylabel('AUC')
            plt.savefig('Pruning_AUC'+str(max(pruned_emb_rate))+'.png')
            plt.clf()

    def post_eval(self,Xi_valid, Xv_valid, y_valid,load_path=None ):
        if load_path is not None:
            print('Evaluate ', load_path,'...')
            ld_model = torch.load(load_path, map_location='cuda:0')
            self.load_state_dict(ld_model)
            model = self.eval()
        else :
            return
        '''
        model_sd = self.state_dict()
        ################################################count FFP8
        all_param_count_exp = []
        total_param_exp_count = [(i,0) for i in range(-140,1)]
        total_max = 0
        total_min = 1
        total_params = 0
        for name in model_sd : 
            if 'top_linear' in name or 'bottom_linear' in name or 'embeddings' in name :
                param = model_sd[name]
                param_count_list = []
                for i in range(-140,1):
                    mask = torch.logical_and((abs(param) > 2.**(i) ) , (abs(param) < 2.**(i+1) ))
                    #print(mask.sum().item())
                    param_count_list.append((i,mask.sum().item()))
                    total_param_exp_count[i+140] = (total_param_exp_count[i+140][0],total_param_exp_count[i+140][1]+mask.sum().item())
                if torch.max(abs(param)) > total_max : 
                    total_max = torch.max(abs(param))
                if torch.min(abs(param)) < total_min :
                    total_min = torch.min(abs(param))
                total_params += np.prod(param.shape)
                layer_dict = {'name':name ,'param_count': param_count_list,'max': torch.max(abs(param)) , 'min' : torch.min(abs(param)), \
                              'total parameters' : np.prod(param.shape) , 'shape' : param.shape}
                all_param_count_exp.append(layer_dict)
        layer_dict = {'name':'All layers' ,'param_count': total_param_exp_count,'max': total_max , 'min' : total_min, \
                              'total parameters' : total_params,'shape' : 0}
        all_param_count_exp.append(layer_dict)
        #############################################################draw_pic
        for layer in all_param_count_exp : 
            print('Layer name',layer['name'])
            print('Param exp count',layer['param_count'])
            print('max_value',layer['max'])
            print('min_value',layer['min'])
            print('Weight shape',layer['shape'])
            print('Total parameters',layer['total parameters'])
            x = [x[0] for x in layer['param_count'] ] 
            y = [x[1] for x in layer['param_count'] ]
            sum_parameters = sum(y)
            y_percentage = [i/sum_parameters for i in y]
            y_percentage_acc = []
            max_exp = math.log(layer['max'].item(),2)
            for i,j in enumerate(y_percentage):
                if i==0 :
                    y_percentage_acc.append(i)
                else:
                    if not (i-140 > max_exp):
                        y_percentage_acc.append(j+y_percentage_acc[i-1])
                    else: 
                        y_percentage_acc.append(0)            
            plt.bar(x,y,zorder=1)
            plt.suptitle('Weight distribution of ' + layer['name'])
            plt.title('max : '+"{:6.3f}".format(layer['max'].item())+' ; exp=' +str(math.floor(max_exp))+';Total parameters='+str(layer['total parameters'].item()))
            plt.xlabel("exp")
            plt.ylabel("# of params")
            plt.scatter(math.floor(max_exp),max(y)*0.05,c="red",zorder=2)
            plt.axvline(x=math.floor(max_exp)-6.5,color='g')
            plt.axvline(x=math.floor(max_exp)-10.5,color='g')
            plt.savefig('exp_cal/'+layer['name']+'.png')
            plt.clf()
            plt.bar(x,y_percentage_acc,zorder=1)
            plt.suptitle('Weight distribution accumulation of ' + layer['name']+ ' by percentage')
            plt.title('max : '+"{:6.3f}".format(layer['max'].item())+' ; exp=' +str(math.floor(max_exp)))
            plt.xlabel("exp")
            plt.ylabel("% accumulation of params")
            plt.scatter(math.floor(max_exp),max(y_percentage_acc)*0.05,c="red",zorder=2)
            plt.axvline(x=math.floor(max_exp)-6.5,color='g')
            plt.axvline(x=math.floor(max_exp)-10.5,color='g')
            #plt.text(10,0,'max : '+"{:6.3f}".format(layer['max'].item())+' ; exp=' +str(math.floor(max_exp)))
            plt.savefig('exp_cal/'+layer['name']+'_percentage_acc.png')
            plt.clf()
            plt.bar(x,y_percentage,zorder=1)
            plt.suptitle('Weight distribution of ' + layer['name']+ ' by percentage')
            plt.title('max : '+"{:6.3f}".format(layer['max'].item())+' ; exp=' +str(math.floor(max_exp)))
            plt.xlabel("exp")
            plt.ylabel("% of params")
            plt.scatter(math.floor(max_exp),max(y_percentage)*0.05,c="red",zorder=2)
            plt.axvline(x=math.floor(max_exp)-6.5,color='g')
            plt.axvline(x=math.floor(max_exp)-10.5,color='g')
            #plt.text(10,0,'max : '+"{:6.3f}".format(layer['max'].item())+' ; exp=' +str(math.floor(max_exp)))
            plt.savefig('exp_cal/'+layer['name']+'_percentage.png')
            plt.clf()
        ########################################################
        '''
        Xi_valid = np.array(Xi_valid).reshape((-1, self.field_size-self.num, 1))
        Xv_valid = np.array(Xv_valid)
        y_valid = np.array(y_valid)
        x_valid_size = Xi_valid.shape[0]
        #is_valid = True
        num_total = non_zero_total = 0
        num_embeddings = non_zero_embeddings = 0
        num_top_mlp = non_zero_top = 0
        num_bot_mlp = non_zero_bot = 0
        
        print('========')
        for name, param in model.named_parameters():
            print (name, param.data.shape)
            num_total += np.prod(param.data.shape)
            non_zero_total += (param != 0).sum().item()
            if 'embeddings' in name:
                num_embeddings += np.prod(param.data.shape)
                non_zero_embeddings += (param != 0).sum().item()
            if 'top_linear' in name:
                num_top_mlp += np.prod(param.data.shape)
                non_zero_top += (param != 0).sum().item()
            if 'bottom_linear' in name:
                num_bot_mlp += np.prod(param.data.shape)
                non_zero_bot += (param != 0).sum().item()
                
        print('Summation of feature sizes: %s' % (sum(self.feature_sizes)))
        print('Number of embeddings: %d' % (num_embeddings))
        print('Number of top MLP parameters: %d' % (num_top_mlp))
        print('Number of bottom MLP parameters: %d' % (num_bot_mlp))
        print("Number of total parameters: %d"% (num_total))
        print('*' * 50)
        print('Number of pruned embeddings: %d, sparse rate %.2f%%' % 
              (non_zero_embeddings, (100 - non_zero_embeddings * 100. / num_embeddings)))
        print('Number of top MLP parameters: %d, sparse rate %.2f%%' % 
              (non_zero_top, (100 - non_zero_top * 100. / num_top_mlp)))
        print('Number of bottom MLP parameters: %d, sparse rate %.2f%%' % 
              (non_zero_bot, (100 - non_zero_bot * 100. / num_bot_mlp)))
        print("Number of pruned total parameters: %d, sparse rate %.2f%%" % 
              (non_zero_total, (100 - non_zero_total * 100. / num_total)))
        print('*' * 50)
        # epoch_begin_time = time()
        valid_result = []
        # self.count_TPR = True
        valid_loss, valid_eval, TPR, TNR, valid_acc, inf_time = self.eval_by_batch(Xi_valid, Xv_valid, y_valid, x_valid_size)
        valid_result.append(valid_eval)
        # self.count_TPR = False
        # print(self.true_pos,self.pos_label,self.true_neg,self.neg_label)
        # print("True positive rate :",(self.true_pos/self.pos_label*100))
        # print("True negative rate :",(self.true_neg/self.neg_label*100))
        # print("Total labels : ",(self.pos_label + self.neg_label))
        # print("Percentage of positive targets :",(self.pos_label/(self.pos_label + self.neg_label)*100))
        print('Validation loss: %.6f AUC: %.6f TPR: %.6f TNR: %.6f Acc: %.6f sparse %.2f%% us/sample: %.3f' %
                  (valid_loss, valid_eval, TPR, TNR, valid_acc, 100 - non_zero_total * 100. / num_total, inf_time))
        print('*' * 50)

    def eval_by_batch(self,Xi, Xv, y, x_size):
        total_loss = 0.0
        y_pred = []
        #if self.use_ffm:
        #    batch_size = 8192*2
        #else:
        batch_size = 8192
        batch_iter = x_size // batch_size
        # print('Test dataset size',x_size)
        criterion = F.binary_cross_entropy_with_logits
        model = self.eval()
        # inferencet time list
        inference_time_list =[]
        
        #start = time()
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
            
            start_time = time()
            outputs = model(batch_xi, batch_xv)
            torch.cuda.synchronize()
            end_time = time()
            if i in range(10, batch_iter):
                inference_time_list.append((end_time - start_time)/batch_size)
            
            pred = torch.sigmoid(outputs).cpu()
            y_pred.extend(pred.data.numpy())
            loss = criterion(outputs, batch_y)
            total_loss += loss.data.item()*(end-offset)

        # if self.use_cuda:
        #     torch.cuda.synchronize()
        # end = time()
        inf_time = (sum(inference_time_list)/len(inference_time_list)) #(end - start)*1e6/x_size
        total_metric = self.eval_metric(y,y_pred)
        total_acc = metrics.accuracy_score(y,np.round(y_pred))
        tn, fp, fn, tp = metrics.confusion_matrix(y, np.round(y_pred)).ravel()
        TPR = tp/(tp+fn)
        TNR = tn/(tn+fp)
        return total_loss/x_size, total_metric, TPR, TNR, total_acc, inf_time*1e6
    
    
    
    

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
            if cnt > 200:
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
# -

'''
if __name__ == '__main__':
    model = DLRM(
        field_size=39,
        feature_sizes=[1458, 556, 245197, 166166, 306, 20, 12055, 634, 4, 46330, 5229, 243454, 3177, 27, 11745, 225322, 11, 4727, 2058, 5, 238640, 18, 16, 67856, 89, 50942], 
        embedding_size=16, 
        n_epochs=20,
        verbose=False, 
        use_cuda=False, 
        batch_size=2048, 
        learning_rate=0.001, 
        weight_decay=6e-7,
        momentum=0,
        sparse=0.9,
        warm=21, 
        h_depth=3,
        numerical=13,
        random_seed=0, 
        top_mlp='512-256', 
        bottom_mlp='512-256-64',
        field_weight=False,
        full_interactions=False, 
        is_bias=True,
        md_sparsity='0.8', 
        deep_layers_activation='relu', 
        interaction_op='mix', 
        compressed_dim=2
        )
    print(model.interaction_op)
    print(model.interaction_size)
    print(model.top_mlp_insize)
    print(model.net_top_linear_0)
'''