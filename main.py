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
Edited by Wang Wei-Lin 
Implement Facebook dlrm to apply structural pruning in DeepFwFM
Reference :
facebookresearch/dlrm
https://github.com/facebookresearch/dlrm
"""
import os,sys,random
from selectors import EpollSelector
import numpy as np
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
from torch.utils.tensorboard import SummaryWriter
import argparse
from utils import data_preprocess
#from FMs import FM
from torchfm.layer import FactorizationMachine, CompressedInteractionNetwork
from torchfm.layer import CrossNetwork
from crossnet import CrossNet
from total_interaction import TotalInteraction
from model.pep_embedding import PEPEmbedding
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

    # if epoch < warmup and warmup != 0:
    #     init_step += 1
    #     lr = (init_lr * init_step) /warmup

    if schedule == 'step' and (epoch > warmup and warmup != 0):
        lr = init_lr * 0.3

    elif schedule == 'cosine' and (epoch > warmup and warmup != 0):
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

def post_eval(eval_only, model, Xi_valid, Xv_valid, y_valid, load_path=None, test_batch_size=8192):
    if load_path is not None:
        print('Evaluate ', load_path,'...')
        ld_model = torch.load(load_path, map_location='cuda:0')
        '''
        # approch 1
        if model.interaction_op == 'mix' and eval_only:
            w1 = ld_model["state_dict"]['catego_mixing.0.weight'].data
            w2T = ld_model["state_dict"]['latent_mixing.0.weight']
            b1 = ld_model["state_dict"]['catego_mixing.0.bias'].data.view(-1, 1).expand(model.num_fea, model.embedding_size)
            b2 = ld_model["state_dict"]['latent_mixing.0.bias'].data.expand(model.num_fea, model.compressed_dim)
            b = (torch.mm(b1, ld_model["state_dict"]['latent_mixing.0.weight'].data.transpose(1,0)) + b2)

            model.w1 = w1.expand(test_batch_size, w1.shape[0], w1.shape[1])
            ld_model["state_dict"]['w2.weight'] = w2T
            model.b = b.expand(test_batch_size, b.shape[0], b.shape[1])
            del ld_model["state_dict"]['catego_mixing.0.weight']
            del ld_model["state_dict"]['catego_mixing.0.bias']
            del ld_model["state_dict"]['latent_mixing.0.weight']
            del ld_model["state_dict"]['latent_mixing.0.bias']

        
        '''
        # approch 2
        if model.interaction_op == 'mix' and eval_only:
            w11 = ld_model["state_dict"]['catego_mixing.0.weight'] #[108, 27]
            b11 = ld_model["state_dict"]['catego_mixing.0.bias'] #[108]
            w12 = ld_model["state_dict"]['catego_mixing.1.weight'] #[27, 108]
            b12 = ld_model["state_dict"]['catego_mixing.1.bias'] #[27]
            w21 = ld_model["state_dict"]['latent_mixing.0.weight'] #[64, 64]
            b21 = ld_model["state_dict"]['latent_mixing.0.bias'] #[64]
            w22 = ld_model["state_dict"]['latent_mixing.1.weight'] #[16, 64]
            b22 = ld_model["state_dict"]['latent_mixing.1.bias'] #[16]
            # w12*w11*x*w21T*w22T + (w12*b11T*(w21T*w22T) + b12*(w21T*w22T) + b21*w22T + b22)
            # = w1*x*w2T + (w12*b11T*w2 + b12*w2 + b21*w22T + b22)
            w1 = torch.mm(w12, w11) #[27, 27]
            w2T = torch.mm(w22, w21) #[16, 64]
            w2 = torch.mm(w21.transpose(0,1), w22.transpose(0,1)) #[64, 16]
            b = torch.mm(torch.mm(w12, b11.view(-1, 1).expand(model.num_fea*4, model.embedding_size)), w2) + torch.mm(b12.view(-1, 1).expand(model.num_fea, model.embedding_size), w2) + torch.mm(b21.expand(model.num_fea, model.embedding_size), w22.transpose(0,1)) + b22.expand(model.num_fea, model.compressed_dim)

            model.w1 = w1.expand(test_batch_size, w1.shape[0], w1.shape[1])
            ld_model["state_dict"]['w2.weight'] = w2T
            model.b = b.expand(test_batch_size, b.shape[0], b.shape[1])
            del ld_model["state_dict"]['catego_mixing.0.weight']
            del ld_model["state_dict"]['catego_mixing.0.bias']
            del ld_model["state_dict"]['catego_mixing.1.weight']
            del ld_model["state_dict"]['catego_mixing.1.bias']
            del ld_model["state_dict"]['latent_mixing.0.weight']
            del ld_model["state_dict"]['latent_mixing.0.bias']
            del ld_model["state_dict"]['latent_mixing.1.weight']
            del ld_model["state_dict"]['latent_mixing.1.bias']
        
        model.load_state_dict(ld_model['state_dict'])
        model = model.eval()
        del ld_model
    else :
        return
    x_valid_size = Xi_valid.shape[0]
    num_total = 0
    num_embeddings = 0
    num_top_mlp = 0
    num_bot_mlp = 0
    print('*' * 50)
    for name, param in model.named_parameters():
        print (name, param.data.shape)
        num_total += np.prod(param.data.shape)
        if 'emb_l' in name:
            num_embeddings += np.prod(param.data.shape)
        if 'top_l' in name:
            num_top_mlp += np.prod(param.data.shape)
        if 'bot_l' in name:
            num_bot_mlp += np.prod(param.data.shape)
            
    print('Summation of feature sizes: %s' % (sum(model.feature_sizes)))
    print('Number of embeddings: %d' % (num_embeddings))
    print('Number of top MLP parameters: %d' % (num_top_mlp))
    print('Number of bottom MLP parameters: %d' % (num_bot_mlp))
    print("Number of total parameters: %d"% (num_total))
    print('*' * 50)

    valid_result = []
    valid_loss, valid_eval, TPR, TNR, valid_acc, inf_time = eval_by_batch(eval_only, model, Xi_valid, Xv_valid, y_valid, x_valid_size, test_batch_size)
    valid_result.append(valid_eval)
    print('Validation loss: %.6f AUC: %.6f TPR: %.6f TNR: %.6f Acc: %.6f Kinf/s: %.3f' %
                (valid_loss, valid_eval, TPR, TNR, valid_acc, inf_time))
    print('*' * 50)

def eval_by_batch(eval_only, model, Xi, Xv, y, x_size, batch_size, use_cuda=True):
    total_loss = 0.0
    y_pred = []
    batch_iter = x_size // batch_size
    criterion = F.binary_cross_entropy_with_logits
    model = model.eval()
    # inferencet time list
    inference_time_list =[]

    for i in range(batch_iter+1):
        offset = i * batch_size
        end = min(x_size, offset + batch_size)
        if offset == end:
            break
        batch_xi = Variable(torch.LongTensor(Xi[offset:end]))
        batch_xv = Variable(torch.FloatTensor(Xv[offset:end]))
        batch_y = Variable(torch.FloatTensor(y[offset:end]))
        if use_cuda:
            batch_xi, batch_xv, batch_y = batch_xi.cuda(), batch_xv.cuda(), batch_y.cuda()
        if end == x_size and model.interaction_op == 'mix' and eval_only:
            bs = batch_xi.shape[0]
            model.w1 = model.w1[:bs]
            model.b = model.b[:bs]
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

    inf_time = len(inference_time_list)/sum(inference_time_list)
    total_metric = roc_auc_score(y,y_pred)
    total_acc = metrics.accuracy_score(y,np.round(y_pred))
    tn, fp, fn, tp = metrics.confusion_matrix(y, np.round(y_pred)).ravel()
    TPR = tp/(tp+fn)
    TNR = tn/(tn+fp)
    return total_loss/x_size, total_metric, TPR, TNR, total_acc, inf_time/1e3

def predict(model, Xi, Xv, use_cuda=True):
    """
    :param Xi: the same as fit function
    :param Xv: the same as fit function
    :return: output, ont-dim array
    """
    Xi = np.array(Xi).reshape((-1, model.field_size - model.num,1))
    Xi = Variable(torch.LongTensor(Xi))
    Xv = Variable(torch.FloatTensor(Xv))
    if use_cuda:
        Xi, Xv = Xi.cuda(), Xv.cuda()

    model = model.eval()
    pred = torch.sigmoid(model(Xi, Xv)).cpu()
    return (pred.data.numpy() > 0.5)


"""
    Network structure
"""
def FeedForward(in_dim, out_dim, dropout=0., activation=nn.Identity()):#nn.ReLU()):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        activation,
        nn.Dropout(dropout)
    )

class Embedding_Linear(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, base_dim, interaction_op):
        super(Embedding_Linear, self).__init__()
        self.interaction_op = interaction_op
        self.embs = nn.EmbeddingBag(
            num_embeddings, embedding_dim, mode="sum", sparse=False)
        torch.nn.init.xavier_uniform_(self.embs.weight)

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
        return self.proj(self.embs(input, offsets=None))

class InteractingLayer(nn.Module):
    """A Layer used in AutoInt that model the correlations between different feature fields by multi-head self-attention mechanism.
      Input shape
            - A 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
            - 3D tensor with shape:``(batch_size,field_size,embedding_size)``.
      Arguments
            - **in_features** : Positive integer, dimensionality of input features.
            - **head_num**: int.The head number in multi-head self-attention network.
            - **use_res**: bool.Whether or not use standard residual connections before output.
            - **seed**: A Python integer to use as random seed.
      References
            - [Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.](https://arxiv.org/abs/1810.11921)
    """

    def __init__(self, embedding_size, head_num=2, use_res=True, scaling=False, seed=1024, device='cpu'):
        super(InteractingLayer, self).__init__()
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        if embedding_size % head_num != 0:
            raise ValueError('embedding_size is not an integer multiple of head_num!')
        self.att_embedding_size = embedding_size // head_num
        self.head_num = head_num
        self.use_res = use_res
        self.scaling = scaling
        self.seed = seed

        self.W_Query = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_key = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Value = nn.Parameter(torch.Tensor(embedding_size, embedding_size))

        if self.use_res:
            self.W_Res = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)

        self.to(device)

    def forward(self, inputs):

        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))

        # None F D
        querys = torch.tensordot(inputs, self.W_Query, dims=([-1], [0]))
        keys = torch.tensordot(inputs, self.W_key, dims=([-1], [0]))
        values = torch.tensordot(inputs, self.W_Value, dims=([-1], [0]))

        # head_num None F D/head_num
        querys = torch.stack(torch.split(querys, self.att_embedding_size, dim=2))
        keys = torch.stack(torch.split(keys, self.att_embedding_size, dim=2))
        values = torch.stack(torch.split(values, self.att_embedding_size, dim=2))

        inner_product = torch.einsum('bnik,bnjk->bnij', querys, keys)  # head_num None F F
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        self.normalized_att_scores = F.softmax(inner_product, dim=-1)  # head_num None F F
        result = torch.matmul(self.normalized_att_scores, values)  # head_num None F D/head_num

        result = torch.cat(torch.split(result, 1, ), dim=-1)
        result = torch.squeeze(result, dim=0)  # None F D
        if self.use_res:
            result += torch.tensordot(inputs, self.W_Res, dims=([-1], [0]))
        result = F.relu(result)

        return result

class DCN(nn.Module):
    def __init__(
        self, field_size, feature_sizes, numerical=13,  device='cuda:0',
        embedding_size=39, fc_sparsity='0.9',
        use_DCNv2 = False, pep_flag=False, opt=None,
    ):
        super(DCN, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.num = numerical

        self.mlp_dims = [1024,1024] if not use_DCNv2 else [512,512,512]
        self.num_layers = 6 if not use_DCNv2 else 3 
        self.device = device
        # Model architecture
        self.embedding_size = embedding_size
        self.fc_sparsity = fc_sparsity
        self.pep_flag = pep_flag

        
        self.num_fea = self.field_size - self.emb_sizes.count(0) - self.num
        self.feat_zero_idx = []
        if self.pep_flag:
            self.emb_l = nn.ModuleList(
                [PEPEmbedding(opt, emb, idx_num) for emb, idx_num in enumerate(self.feature_sizes)]
            )
        elif self.fc_sparsity == '0':
            self.emb_l = nn.ModuleList(
                [nn.EmbeddingBag(n, m, mode="sum", sparse=False) for n , m in zip(self.feature_sizes, self.emb_sizes)]
            )
        else:
            self.emb_l = nn.ModuleList()
            for i in range(len(self.feature_sizes)):
                n = self.feature_sizes[i]
                m = self.emb_sizes[i]
                if m==0:
                    self.feat_zero_idx.append(i)
                    continue
                EE = nn.EmbeddingBag(n, m, mode="sum", sparse=False) 
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                EE.weight.data = torch.tensor(W, requires_grad=True)
                self.emb_l.append(EE)
                    
        self.embed_output_dim = sum(self.emb_sizes) + self.num
        self.mlp_dims = [self.embed_output_dim]+self.mlp_dims
        if use_DCNv2:
            self.cn = CrossNet(self.embed_output_dim, layer_num=self.num_layers, parameterization='matrix', device=self.device)
        else:
            # self.cn = CrossNetwork(self.embed_output_dim, self.num_layers)
            self.cn = CrossNet(self.embed_output_dim, layer_num=self.num_layers, parameterization='vector', device=self.device)
        self.deep_l = nn.Sequential(
            *[FeedForward(self.mlp_dims[i], self.mlp_dims[i+1], 0.0, nn.ReLU()) for i in range(len(self.mlp_dims)-1)]
        )

        self.linear = torch.nn.Linear(self.mlp_dims[-1] + self.embed_output_dim, 1)
        
    def forward(self, Xi, Xv):
        """
        :param Xi_train: index input tensor, batch_size * embedding_size * 1
        :return: the last output
        """

        # Embedding
        emb_out = []        
        k=0
        for j in range(len(self.feature_sizes)):
            if j in self.feat_zero_idx:
                continue
            V = self.emb_l[k](Xi[:,j,:])
            emb_out.append(V)
            k+=1
        emb_output = torch.cat((Xv, torch.cat(emb_out,1)), 1)
        # Interaction
        cross_output = self.cn(emb_output)
        
        #Deep part
        dnn_output = self.deep_l(emb_output)
        
        #sum
        return self.linear(torch.cat((dnn_output, cross_output), 1)).squeeze(1)        

class AutoInt(nn.Module):
    def __init__(
        self, field_size, feature_sizes, numerical=13, device='cuda:0',
        embedding_size=16, fc_sparsity='0.8', pep_flag=False, opt=None,
    ):
        super(AutoInt, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.num = numerical
        self.device = device
        self.fc_sparsity = fc_sparsity
        self.dropout_deep = [0.5,0.5,0.5,0.5]
        self.num_deeps = 1
        self.deep_layers = [400,400,400]
        self.deep_nodes = 400
        self.dropout_shallow = [0.1,0.1]
        self.att_layer_num=3
        self.att_embedding_size=32
        self.att_head_num=2
        self.att_res=True
        self.dnn_hidden_units=(256, 128)
        self.dnn_activation='relu'
        self.l2_reg_linear=1e-5
        self.l2_reg_embedding=1e-5
        self.l2_reg_dnn=0
        self.dnn_use_bn=False
        self.dnn_dropout=0

        dnn_feature_columns = [embedding_size] *self.num
        self.atten_dim = 32
        
        self.use_dnn = len(dnn_feature_columns) > 0 and len(self.dnn_hidden_units) > 0
        field_num = field_size
                
        if len(self.dnn_hidden_units) <= 0 and self.att_layer_num <= 0:
            print("Either hidden_layer or att_layer_num must > 0")
            exit(1)
                
        # Model architecture
        self.embedding_size = embedding_size
        self.pep_flag = pep_flag

        self.feature_sizes = [1]*self.num + list(self.feature_sizes)
        if fc_sparsity == '0':
            self.emb_sizes = [embedding_size] * len(self.feature_sizes)
        else:
            self.emb_sizes = emb_dict[embedding_size][fc_sparsity]
            # self.emb_sizes = [embedding_size]* self.num + self.emb_sizes
        
        self.num_fea = self.field_size - self.emb_sizes.count(0) - self.num
        self.feat_zero_idx = []
        if self.pep_flag:
            self.emb_l = nn.ModuleList(
                [PEPEmbedding(opt, emb, idx_num) for emb, idx_num in enumerate(self.feature_sizes)]
            )
        elif self.fc_sparsity == '0':
            self.emb_l = nn.ModuleList(
                [nn.EmbeddingBag(n, m, mode="sum", sparse=False) for n , m in zip(self.feature_sizes, self.emb_sizes)]
            )
        else:
            self.emb_l = nn.ModuleList()
            for i in range(len(self.feature_sizes)):
                n = self.feature_sizes[i]
                m = self.emb_sizes[i]
                if m==0:
                    self.feat_zero_idx.append(i)
                    continue
                EE = nn.EmbeddingBag(n, m, mode="sum", sparse=False)
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                EE.weight.data = torch.tensor(W, requires_grad=True)
                self.emb_l.append(EE)
        """
        Initial AutoInt Part
        """
        
        if len(self.dnn_hidden_units) and self.att_layer_num > 0:
            dnn_linear_in_feature = self.dnn_hidden_units[-1] + field_num * self.atten_dim
        elif len(self.dnn_hidden_units) > 0:
            dnn_linear_in_feature = self.dnn_hidden_units[-1]
        elif self.att_layer_num > 0:
            dnn_linear_in_feature = field_num * self.atten_dim
        else:
            raise NotImplementedError
        self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False)
        if self.use_dnn:
            
            inputs_dim = sum(self.emb_sizes)
            hidden_units = self.dnn_hidden_units
            hidden_units = [inputs_dim] + list(hidden_units)
            self.deep_l = nn.Sequential(
                *[FeedForward(hidden_units[i], hidden_units[i+1], 0.0, nn.ReLU()) for i in range(len(hidden_units)-1)]
            )


            # self.dnn_linears = nn.ModuleList(
            # [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])
            # self.dnn_activation_layers = nn.ModuleList(
            # [nn.ReLU() for i in range(len(hidden_units) - 1)])
            #self.add_regularization_weight(
            #filter(lambda x: 'dnn_' in x[0] and 'weight' in x[0] and 'bn' not in x[0], self.named_parameters()), l2=self.l2_reg_dnn)
        self.atten_embedding = nn.Linear(embedding_size, self.atten_dim)#, bias=False
        # self.atten_embedding = nn.ModuleList(
        #         [nn.Linear(m, self.atten_dim, bias=False) for m in self.emb_sizes]
        # )
        self.int_layers = nn.Sequential(
            *[InteractingLayer(self.atten_dim, self.att_head_num, self.att_res, device=self.device) for _ in range(self.att_layer_num)]
        )
        # self.AutoInt_embeddings = nn.ModuleList(
        #     [nn.Embedding(feature_size, embedding_size) for feature_size , embedding_size in zip(self.feature_sizes,self.emb_sizes)])
        
 
    def forward(self, Xi, Xv):
        """
        :param Xi_train: index input tensor, batch_size * embedding_size * 1
        :return: the last output
        """
        Tzero = torch.zeros(Xi.shape[0], 1, dtype=torch.long).to(self.device)
        # if use_cuda:
        #     Tzero = Tzero.cuda()
        
        # Embedding
        emb_out = []
        dnn_input=[]        
        k=0
        for i, emb in enumerate(self.emb_l):
            if i in self.feat_zero_idx:
                continue
            if i < self.num:
                V = self.emb_l[k](Tzero) * Xv[:,i].unsqueeze(1)
            else:
                V = self.emb_l[k](Xi[:,i-self.num,:])
            dnn_input.append(V)
            
            V = F.pad(V, (0, self.embedding_size - V.size(1))) ## pad
            emb_out.append(V)
            k+=1
        
        # Interaction
        att_input = self.atten_embedding(torch.stack(emb_out,1))
        # att_input=[]
        # for i, proj in enumerate(self.atten_embedding):
        #     att_input.append(proj(emb_out[i]))
        # att_input = torch.stack(att_input, 1)
        att_input = self.int_layers(att_input)
        att_output = torch.flatten(att_input, start_dim=1) 
        
        #Deep part          
        dnn_output = self.deep_l(torch.cat(dnn_input,1))
        # for i in range(len(self.dnn_linears)):
        #     fc = self.dnn_linears[i](dnn_input)
        #     dnn_input = self.dnn_activation_layers[i](fc)                    
        #sum
        return self.dnn_linear(torch.cat((dnn_output, att_output), 1)).squeeze(1)        

class FM(nn.Module):
    def __init__(
        self, field_size, feature_sizes, numerical=13,
        embedding_size=64, fc_sparsity='0.6', pep_flag=False, opt=None,
    ):
        super(FM, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.num = numerical
        # Model architecture
        self.embedding_size = embedding_size
        self.bias = torch.nn.Parameter(torch.Tensor([0.01]))
        self.fc_sparsity = fc_sparsity
        self.pep_flag = pep_flag
        
        self.num_fea = self.field_size - self.emb_sizes.count(0)

        """
        Initial FM part
        """
        self.fm = FactorizationMachine(reduce_sum=False)
        self.w_l = nn.ModuleList([nn.EmbeddingBag(n, 1, mode="sum", sparse=False) for n in self.feature_sizes])
        self.feat_zero_idx=[]
        if self.pep_flag:
            self.emb_l = nn.ModuleList(
                [PEPEmbedding(opt, emb, idx_num) for emb, idx_num in enumerate(self.feature_sizes)]
            )
        elif self.fc_sparsity == '0':
            self.emb_l = nn.ModuleList(
                [nn.EmbeddingBag(n, m, mode="sum", sparse=False) for n , m in zip(self.feature_sizes,self.emb_sizes)]
            )
        else:
            self.emb_l = nn.ModuleList()
            # 2022/06/13 testing the speed of linear transformation vs. padding
            # self.linear_transformation = nn.ModuleList()
            for i in range(len(self.feature_sizes)):
                n = self.feature_sizes[i]
                m = self.emb_sizes[i]
                if m==0:
                    self.feat_zero_idx.append(i)
                    continue
                EE = nn.EmbeddingBag(n, m, mode="sum", sparse=False) 
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                EE.weight.data = torch.tensor(W, requires_grad=True)
                self.emb_l.append(EE)
                # self.linear_transformation.append(nn.Linear(m, embedding_size, bias=False))        

    def forward(self, Xi):
        """
        :param Xi_train: index input tensor, batch_size * embedding_size * 1
        :return: the last output
        """
        # Embedding
        linear_out = []
        for i, emb in enumerate(self.w_l):
            linear_out.append(emb(Xi[:,i,:]))
        fm_first_order = torch.cat(linear_out, 1)
            
        emb_out = []        
        k=0
        for j in range(len(self.feature_sizes)):
            if j in self.feat_zero_idx:
                continue
            
            V = self.emb_l[k](Xi[:,j,:])
            V = F.pad(V, (0, self.embedding_size - V.size(1))) # torch.Size([2048, 10])
            # 2022/06/13 testing the speed of linear transformation vs. padding
            # V = self.linear_transformation[k](V)
            emb_out.append(V)
            k+=1
        fm_second_order = self.fm(torch.stack(emb_out,1))

        return torch.sum(fm_first_order,1) + torch.sum(fm_second_order,1) + self.bias

class DeepFM(nn.Module):
    def __init__(
        self, field_size, feature_sizes, numerical=13,
        embedding_size=64, fc_sparsity='0.6', pep_flag=False, opt=None,
    ):
        super(DeepFM, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.num = numerical
        print(feature_sizes)
        print(len(feature_sizes))
        # Model architecture
        self.embedding_size = embedding_size
        self.bias = torch.nn.Parameter(torch.Tensor([0.01]))
        self.fc_sparsity = fc_sparsity
        self.pep_flag = pep_flag

        self.num_fea = self.field_size - self.emb_sizes.count(0)
        self.deep_layers = [self.num_fea*self.embedding_size] + [400, 400, 400]

        """
        Initial FM part
        """
        self.fm = FactorizationMachine(reduce_sum=False)
        self.w_l = nn.ModuleList([nn.EmbeddingBag(n, 1, mode="sum", sparse=False) for n in self.feature_sizes])
        self.feat_zero_idx=[]
        if self.pep_flag:
            self.emb_l = nn.ModuleList(
                [PEPEmbedding(opt, emb, idx_num) for emb, idx_num in enumerate(self.feature_sizes)]
            )
        elif self.fc_sparsity == '0':
            self.emb_l = nn.ModuleList(
                [nn.EmbeddingBag(n, m, mode="sum", sparse=False) for n , m in zip(self.feature_sizes, self.emb_sizes)]
            )
        else:
            self.emb_l = nn.ModuleList()
            for i in range(len(self.feature_sizes)):
                n = self.feature_sizes[i]
                m = self.emb_sizes[i]
                if m==0:
                    self.feat_zero_idx.append(i)
                    continue
                EE = nn.EmbeddingBag(n, m, mode="sum", sparse=False) 
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                EE.weight.data = torch.tensor(W, requires_grad=True)
                self.emb_l.append(EE)

        """
        Initial Deep part
        """
        self.deep_l = nn.Sequential(
            *[FeedForward(self.deep_layers[i], self.deep_layers[i+1], 0.5, nn.ReLU()) for i, d in enumerate(self.deep_layers[:-1])],
            nn.Linear(self.deep_layers[-1], 1),
        )

    def forward(self, Xi):
        """
        :param Xi_train: index input tensor, batch_size * embedding_size * 1
        :return: the last output
        """
        # Embedding
        linear_out = []
        for i, emb in enumerate(self.w_l):
            linear_out.append(emb(Xi[:,i,:]))
        fm_first_order = torch.cat(linear_out, 1)
            
        emb_out = []        
        k=0
        for j in range(len(self.feature_sizes)):
            if j in self.feat_zero_idx:
                continue
            
            V = self.emb_l[k](Xi[:,j,:])
            V = F.pad(V, (0, self.embedding_size - V.size(1))) # torch.Size([2048, 10])
            emb_out.append(V)
            k+=1
        fm_second_order = self.fm(torch.stack(emb_out,1))

        deep_out = self.deep_l(torch.cat(emb_out, 1))

        return torch.sum(fm_first_order,1) + torch.sum(fm_second_order,1) + torch.sum(deep_out,1) + self.bias

class xDeepFM(nn.Module):
    def __init__(
        self, field_size, feature_sizes, numerical=13,
        embedding_size=64, fc_sparsity='0.6', pep_flag=False, opt=None,
    ):
        super(xDeepFM, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.num = numerical

        # Model architecture
        self.embedding_size = embedding_size
        self.cross_layer_sizes = [200, 200, 200]
        self.bias = torch.nn.Parameter(torch.Tensor([0.01]))
        self.fc_sparsity = fc_sparsity
        self.pep_flag = pep_flag
        
        self.num_fea = self.field_size - self.emb_sizes.count(0)
        self.deep_layers = [self.num_fea*self.embedding_size] + [400, 400]

        """
        Initial FM part
        """
        self.cin = CompressedInteractionNetwork(self.num_fea, self.cross_layer_sizes, split_half=True)
        self.w_l = nn.ModuleList([nn.EmbeddingBag(n, 1, mode="sum", sparse=False)  for n in self.feature_sizes])
        self.feat_zero_idx=[]
        if self.pep_flag:
            self.emb_l = nn.ModuleList(
                [PEPEmbedding(opt, emb, idx_num) for emb, idx_num in enumerate(self.feature_sizes)]
            )
        elif self.fc_sparsity == '0':
            self.emb_l = nn.ModuleList(
                [nn.EmbeddingBag(n, m, mode="sum", sparse=False) for n , m in zip(self.feature_sizes,self.emb_sizes)]
            )
        else:
            self.emb_l = nn.ModuleList()
            for i in range(len(self.feature_sizes)):
                n = self.feature_sizes[i]
                m = self.emb_sizes[i]
                if m==0:
                    self.feat_zero_idx.append(i)
                    continue
                EE = nn.EmbeddingBag(n, m, mode="sum", sparse=False) 
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                EE.weight.data = torch.tensor(W, requires_grad=True)
                self.emb_l.append(EE)

        """
        Initial Deep part
        """
        self.deep_l = nn.Sequential(
            *[FeedForward(self.deep_layers[i], self.deep_layers[i+1], 0.0, nn.ReLU()) for i, d in enumerate(self.deep_layers[:-1])],
            nn.Linear(self.deep_layers[-1], 1, bias=False),
        )

    def forward(self, Xi):
        """
        :param Xi_train: index input tensor, batch_size * embedding_size * 1
        :return: the last output
        """
        
        # Embedding
        linear_out = []
        for i, emb in enumerate(self.w_l):
            linear_out.append(emb(Xi[:,i,:]))
        fm_first_order = torch.cat(linear_out, 1)
            
        emb_out = []        
        k=0
        for j in range(len(self.feature_sizes)):
            if j in self.feat_zero_idx:
                continue
            
            V = self.emb_l[k](Xi[:,j,:])
            V = F.pad(V, (0, self.embedding_size - V.size(1))) # torch.Size([2048, 10])
            emb_out.append(V)
            k+=1
        fm_second_order = self.cin(torch.stack(emb_out,1)) #torch.Size([2048, 1])

        deep_out = self.deep_l(torch.cat(emb_out, 1))

        return torch.sum(fm_first_order,1) + torch.sum(fm_second_order,1) + torch.sum(deep_out,1) + self.bias

class DeepFwFM(nn.Module):
    def __init__(
        self, field_size, feature_sizes, numerical=13, device='cuda:0',
        embedding_size=10, fc_sparsity='0.6', pep_flag=False, opt=None,
    ):
        super(DeepFwFM, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.num = numerical
        self.device = device
        # Model architecture
        self.embedding_size = embedding_size
        self.bias = torch.nn.Parameter(torch.Tensor([0.01]))
        self.fc_sparsity = fc_sparsity
        self.pep_flag = pep_flag

        self.feature_sizes = [1]*self.num + self.feature_sizes
        if self.fc_sparsity == '0':
            self.emb_sizes = [self.embedding_size] * len(self.feature_sizes)
        else:
            self.emb_sizes = emb_dict[self.embedding_size][self.fc_sparsity]
            # self.emb_sizes = [self.embedding_size]* self.num + self.emb_sizes
        
        self.num_fea = self.field_size - self.emb_sizes.count(0)
        self.deep_layers = [self.num_fea*self.embedding_size] + [400, 400]

        """
        Initial FwFM part
        """
        self.fm_first_order_dropout = nn.Dropout(0.5)
        self.feat_zero_idx=[]
        if self.pep_flag:
            self.emb_l = nn.ModuleList(
                [PEPEmbedding(opt, emb, idx_num) for emb, idx_num in enumerate(self.feature_sizes)]
            )
        elif self.fc_sparsity == '0':
            self.emb_l = nn.ModuleList(
                [nn.EmbeddingBag(n, m, mode="sum", sparse=False) for n , m in zip(self.feature_sizes,self.emb_sizes)]
            )
        else:
            self.emb_l = nn.ModuleList()
            for i in range(len(self.feature_sizes)):
                n = self.feature_sizes[i]
                m = self.emb_sizes[i]
                if m==0:
                    self.feat_zero_idx.append(i)
                    continue
                EE = nn.EmbeddingBag(n, m, mode="sum", sparse=False) 
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                EE.weight.data = torch.tensor(W, requires_grad=True)
                self.emb_l.append(EE)
        self.fm_second_order_dropout = nn.Dropout(0.1)
        self.fwfm_linear = nn.Linear(self.embedding_size, self.num_fea, bias=False)
        self.field_cov = nn.Linear(self.num_fea, self.num_fea, bias=False)
        """
        Initial Deep part
        """
        self.deep_l = nn.Sequential(
            nn.Dropout(0.5),
            *[FeedForward(self.deep_layers[i], self.deep_layers[i+1], 0.5, nn.ReLU()) for i, d in enumerate(self.deep_layers[:-1])],
            nn.Linear(self.deep_layers[-1], 1, bias=False),
        )

    def forward(self, Xi, Xv):
        """
        :param Xi_train: index input tensor, batch_size * embedding_size * 1
        :return: the last output
        """
        Tzero = torch.zeros(Xi.shape[0], 1, dtype=torch.long).to(self.device)
        
        # Embedding
        emb_out = []
        k=0
        for i in range(len(self.feature_sizes)):
            if i in self.feat_zero_idx:
                continue
            if i < self.num:
                V = self.emb_l[k](Tzero) * Xv[:,i].unsqueeze(1)
            else:
                V = self.emb_l[k](Xi[:,i-self.num,:])
        
            V = F.pad(V, (0, self.embedding_size - V.size(1))) ## pad
            emb_out.append(V)
            k+=1
        fm_second_order_tensor = torch.stack(emb_out)
        fwfm_linear = torch.einsum('ijk,ik->ijk', [fm_second_order_tensor, self.fwfm_linear.weight])
        fm_first_order = torch.einsum('ijk->ji', [fwfm_linear])
        fm_first_order = self.fm_first_order_dropout(fm_first_order)

        # Interaction
        outer_fm = torch.einsum('kij,lij->klij', fm_second_order_tensor, fm_second_order_tensor)
        outer_fwfm = torch.einsum('klij,kl->klij', outer_fm, (self.field_cov.weight.t() + self.field_cov.weight) * 0.5)
        fm_second_order = (torch.sum(torch.sum(outer_fwfm, 0), 0) - torch.sum(torch.einsum('kkij->kij', outer_fwfm), 0)) * 0.5
        fm_second_order = self.fm_second_order_dropout(fm_second_order)

        deep_out = self.deep_l(torch.cat(emb_out, 1))           

        return torch.sum(fm_first_order,1) + torch.sum(fm_second_order,1) + torch.sum(deep_out,1) + self.bias

class DLRM(nn.Module):

    def __init__(
        self, eval_only, field_size, feature_sizes, numerical=13,
        embedding_size=64, fc_sparsity='0.8', bottom_mlp='13-512-256-64', top_mlp='512-256',
        interaction_op='gate', full_interactions=False, compressed_dim=8, 
        expansion_factor='4-1', mix_act='relu', mix_residual=False, device='cuda:0', test_batch_size=131768,
    ):
        super(DLRM, self).__init__()
        self.field_size = field_size
        self.feature_sizes = feature_sizes
        self.num = numerical
        self.pep_flag = False
        
        # Model architecture
        self.embedding_size = embedding_size

        self.fc_sparsity = fc_sparsity
        if fc_sparsity == '0':
            self.emb_sizes = [embedding_size] * len(self.feature_sizes)
        else:
            self.emb_sizes = emb_dict[embedding_size][fc_sparsity]
        self.interaction_op = interaction_op
        self.num_fea = self.field_size - self.emb_sizes.count(0) - self.num + 1

        if self.interaction_op=='dot' or self.interaction_op=='paddot':
            if full_interactions:
                self.interaction_size = (self.num_fea)**2
            else:
                self.interaction_size = int(self.num_fea * (self.num_fea - 1) / 2)
        elif self.interaction_op=='cat' or self.interaction_op=='gate':
            self.interaction_size = sum(self.emb_sizes)
        elif self.interaction_op=='mix':
            self.expansion_factor = np.fromstring(expansion_factor, dtype=int, sep="-")
            self.eval_only=eval_only
            self.mix_act = mix_act
            self.mix_residual = mix_residual
            self.device = device
            self.test_batch_size = test_batch_size
            self.compressed_dim = compressed_dim
            self.interaction_size = self.num_fea * self.compressed_dim
        else:
            sys.exit(
                "ERROR: --interaction_op="
                + interaction_op
                + " is not supported"
            )
        self.bottom_mlp = np.fromstring(bottom_mlp, dtype=int, sep="-") # "13-512-256-64"
        self.bottom_mlp[0] = self.num
        self.top_mlp_insize = self.interaction_size + self.bottom_mlp[-1]*2
        top_mlp_adjusted = str(self.top_mlp_insize) + "-" + top_mlp
        self.top_mlp = np.fromstring(top_mlp_adjusted, dtype=int, sep="-") # top_mlp_insize + "512-256"

        """
            Embedding
        """
        self.emb_l = nn.ModuleList()
        self.feat_zero_idx=[]
        for i in range(len(self.feature_sizes)):
            n = self.feature_sizes[i]
            m = self.emb_sizes[i]
            if m==0:
                self.feat_zero_idx.append(i)
                continue
            if self.interaction_op=='dot' and fc_sparsity!='0':
                EE = Embedding_Linear(n, m, self.embedding_size, self.interaction_op)
                # use np initialization as below for consistency...
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                EE.embs.weight.data = torch.tensor(W, requires_grad=True)
            else:
                EE = nn.EmbeddingBag(n, m, mode="sum", sparse=False) 
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                EE.weight.data = torch.tensor(W, requires_grad=True)
            
            self.emb_l.append(EE)
            
        """
            Interaction
        """
        if self.interaction_op=='gate':
            self.gating = nn.Sequential(
                nn.BatchNorm1d(self.top_mlp_insize, momentum=0.005),
                nn.Linear(self.top_mlp_insize, self.top_mlp_insize),
                #nn.LayerNorm(self.top_mlp_insize),
                nn.Sigmoid()
            )

        elif self.interaction_op=='mix':
        
            self.ti = TotalInteraction(
                self.num_fea, self.embedding_size, self.compressed_dim, \
                self.expansion_factor, self.mix_act, self.mix_residual, \
                self.device, self.eval_only, self.test_batch_size
            )
            
        """
            Deep parts
        """
        self.bot_l = nn.Sequential(
            *[FeedForward(self.bottom_mlp[i], self.bottom_mlp[i+1], 0.5, nn.ReLU()) for i, d in enumerate(self.bottom_mlp[:-1])]
        )
        ## Multi-Bottom
        self.bot_l_1 = nn.Sequential(
            *[FeedForward(self.bottom_mlp[i], self.bottom_mlp[i+1], 0.5, nn.ReLU()) for i, d in enumerate(self.bottom_mlp[:-1])]
        )
        self.bot_l_2 = nn.Sequential(
            *[FeedForward(self.bottom_mlp[i], self.bottom_mlp[i+1], 0.5, nn.ReLU()) for i, d in enumerate(self.bottom_mlp[:-1])]
        )
        ##
        self.top_l = nn.Sequential(
            *[FeedForward(self.top_mlp[i], self.top_mlp[i+1], 0.5, nn.ReLU()) for i, d in enumerate(self.top_mlp[:-1])],
            nn.Linear(self.top_mlp[-1], 1)#, nn.Dropout(0.5)
        )
        
        ## Bidirectional Fusion
        self.DNN_1 = FeedForward(26, 26, 0.5)
        self.DNN_2 = FeedForward(26, 2, 0.5)
        ##    
        
        ##dense weight
        self.dense_weight = nn.Parameter(torch.ones(13))
        self.dense_weight_1 = nn.Parameter(torch.ones(13))
        self.dense_weight_2 = nn.Parameter(torch.ones(13))
        ##

    def forward(self, Xi, Xv):

        # Instance Bis
        #Xv = Xv + self.instance_bias
        
        # Dense Weight
        Xv_0 = Xv * self.dense_weight #+ self.dense_bias
        Xv_1 = Xv * self.dense_weight_1 #+ self.dense_bias_1
        Xv_2 = Xv * self.dense_weight_2 #+ self.dense_bias_2
        
        # Bottom MLP part
        bottom_mlp_result = self.bot_l(Xv_0) # 2048*16
        
        # Multi-Bottom 
        bottom_mlp_result_1 = self.bot_l_1(Xv_1)
        bottom_mlp_result_2 = self.bot_l_2(Xv_2)
        #
        
        # Embedding
        k=0
        embedding_array=[]
        for j in range(len(self.feature_sizes)):
            if j in self.feat_zero_idx:
                continue
            V = self.emb_l[k](Xi[:,j,:])

            if self.interaction_op=='mix' or self.interaction_op=='paddot':
                V = F.pad(V, (0, self.embedding_size - V.size(1)))

            embedding_array.append(V)
            k+=1

        # Interaction
        if self.interaction_op=='dot' or self.interaction_op=='paddot':
            dlrm_emb_vec = torch.stack(embedding_array)
            (batch_size, d) = embedding_array[0].shape # 26*2048*64
            interaction_vec = torch.stack((embedding_array + [bottom_mlp_result]), dim=1) # 27*2048*64 -> 2048*27*64 # .view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(interaction_vec, torch.transpose(interaction_vec, 1, 2)) # 2048*27*64 * 2048*64*27 -> 2048*27*27
            _, ni, nj = Z.shape
            offset = 0 # offset = 1 if full_interactions else 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            top_mlp_input = torch.cat(([Zflat] + [bottom_mlp_result]), dim=1) # R = torch.cat([x] + [Zflat], dim=1)

        elif self.interaction_op=='cat':
            dlrm_emb_vec = torch.cat(embedding_array,1)
            top_mlp_input = torch.cat((dlrm_emb_vec,bottom_mlp_result), 1)
            
        elif self.interaction_op=='gate':
            emb_vec = torch.cat(embedding_array,1)
            interaction_vec = torch.cat((emb_vec,bottom_mlp_result), 1)
            top_mlp_input = self.gating(interaction_vec) * interaction_vec
        
        elif self.interaction_op=='mix':
            #bottom_mlp_result = self.MIX_LN2(bottom_mlp_result)
            if not self.eval_only:
                dlrm_emb_vec = torch.stack((embedding_array + [bottom_mlp_result]), 2)
            else:
                dlrm_emb_vec = torch.stack((embedding_array + [bottom_mlp_result]), 1)
            
            ################ Birdirectional Fusion ####################
            dense_stack = torch.stack(embedding_array, 2) #(B, 64, 26)
            dense_stack = self.DNN_1(dense_stack)         #(B, 64, 26)
            dense_stack = self.DNN_2(dense_stack)         #(B, 64, 1 )
            dense_stack = dense_stack.transpose(0,2)      #(1, 64, B)
            dense_stack = dense_stack.transpose(1,2)      #(1, B, 64)            
            ##
            dense_1 = dense_stack[0] * bottom_mlp_result_1
            dense_2 = dense_stack[1] * bottom_mlp_result_2
            ##
            bottom_mlp_result = torch.cat((dense_1, dense_2),1)
            ###########################################################
        
            top_mlp_input = torch.cat((self.ti(dlrm_emb_vec), bottom_mlp_result), 1)    
        # Top MLP part
        top_mlp_result = self.top_l(top_mlp_input)
        result = top_mlp_result.squeeze()
        return result


# credit to https://github.com/ChenglongChen/tensorflow-DeepFM/blob/master/DeepFM.py
def init_weights(model, use_cuda=True):
    TORCH = torch.cuda if use_cuda else torch
    for name, param in model.named_parameters():
        if 'w_l' in name:
            param.data = TORCH.FloatTensor(param.data.size()).normal_()
        elif 'emb_l' in name and model.fc_sparsity=='0' and not model.pep_flag:
            param.data = TORCH.FloatTensor(param.data.size()).normal_().mul(0.01)
        elif 'deep_l' in name:
            if 'weight' in name: # weight and bias in the same layer share the same glorot
                glorot =  np.sqrt(2.0 / np.sum(param.data.shape))
            param.data = TORCH.FloatTensor(param.data.size()).normal_().mul(glorot)
        elif 'field_cov.weight' == name:
            param.data = TORCH.FloatTensor(param.data.size()).normal_().mul(np.sqrt(2.0 / model.field_size / 2))

def calc_sparsity(model):
    sparsity_l=[]
    params_l=[]
    total_params=0
    for i in range(len(model.emb_l)):
        non_zero_values = torch.nonzero(model.emb_l[i].sparse_v, as_tuple=False).size(0)
        base = torch.numel(model.emb_l[i].sparse_v)
        sparsity = 1 - (non_zero_values / base)
        sparsity_l.append(round(sparsity, 3))
        params_l.append(non_zero_values)
        total_params += non_zero_values
    percentage = 1 - (sum(params_l) / (sum(model.feature_sizes) * model.embedding_size))
    return sparsity_l, params_l, round(percentage, 4), total_params

def get_threshold(model):
    thr_list=[]
    for i in range(26):
        thr_list.append(torch.sigmoid(model.emb_l[i].s))
    return thr_list

def save_pruned_embedding(model, sparsity_l, params_l, percentage, total_params):

    max_candidate_p = max(candidate_p)
    if max_candidate_p == 0:
        print("Minimal target parameters achieved, stop pruning.")
        exit(0)
    else:
        if total_params <= max_candidate_p:
            # emb_save_path = opt['emb_save_path'].format(overall_sparsity=percentage, embid='{embid}') #'./tmp/embedding/{task}/{overall_sparsity}/{embid}.npy'
            emb_save_path = opt['emb_save_path'].format(sparsity=percentage, total_params=total_params) #'./tmp/embedding/{task}/sparsity_{sparsity}_total_params_{total_params}.txt'
            emb_save_dir, _ = os.path.split(emb_save_path) #'./tmp/embedding/{task}/ ##{overall_sparsity}/'
            if not os.path.exists(emb_save_dir):
                os.makedirs(emb_save_dir)
            
            emb_dims = np.round(model.embedding_size * (1 - np.array(sparsity_l)))
            # with open(emb_save_dir + '/sparsity_{}_total_params_{}.txt'.format(percentage, total_params), 'w') as f:
            with open(emb_save_path, 'w') as f:
                f.write('embedding dimension: ' + str(emb_dims) + '\n')
                f.write('sparsity_l: ' + str(sparsity_l) + '\n')
                f.write('params_l: ' + str(params_l) + '\n')
            # for i in range(len(model.emb_l)):
            #     embedding = model.emb_l[i].sparse_v.detach().cpu().numpy()
            #     np.save(emb_save_path.format(embid=i), embedding)
                
            max_idx = candidate_p.index(max(candidate_p))
            candidate_p[max_idx] = 0
            print("*" * 80)
            print("Reach the target parameter: {}, save embedding with size: {}".format(max_candidate_p, total_params))
            print("*" * 80)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter tuning')
    parser.add_argument('--use_cuda', action='store_true', default=True)
    parser.add_argument('--gpu', default=0, type=int, help='GPU id')
    parser.add_argument('--numerical', default=13, type=int, help='Numerical features, 13 for Criteo')
    parser.add_argument('--n_epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--learning_rate', default= 0.001, type=float)
    parser.add_argument('--warm', type=int, default=20)
    parser.add_argument("--lr_decay", type=str, default="none", choices=["step", "cosine", "none"])
    parser.add_argument("--lr_alpha",  type=float, default=0.0)
    parser.add_argument('--momentum', default= 0, type=float)
    parser.add_argument('--optimizer_type', type=str, default="adam", choices=["sgd", "adam", "rmsp", "adag"])
    parser.add_argument('--weight_decay', default=6e-7, type=float, help='L2 penalty') # 3e-7
    parser.add_argument('--embedding_size', default=64, type=int)
    parser.add_argument('--bottom_mlp', default='13-512-256-64')
    parser.add_argument('--top_mlp', default='512-256', help='exclude last layer output = 1')
    parser.add_argument('--interaction_op', default='dot', type=str, choices=["dot", "paddot", "cat", "gate", "mix"])
    parser.add_argument('--full_interactions', action='store_true', default=False, help='Use 27*27 interactions result or 27*26/2')
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--fc_sparsity", type=str, default='0')
    parser.add_argument("--compressed_dim", type=int ,default=16)
    parser.add_argument("--expansion_factor", type=str, default='4-1')
    parser.add_argument("--mix_act", type=str, default='relu')
    parser.add_argument("--mix_residual", action='store_true', default=False)
    parser.add_argument('--eval_only', action='store_true', default=False, help='Run with only evaluate')
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--save_path', default=None, type=str)
    parser.add_argument('--test_batch_size', type=int ,default=131072) #8192
    parser.add_argument('--model', type=str, default='DLRM', choices=['FM', 'DeepFM', 'xDeepFM', 'DeepFwFM', 'AutoInt', 'DCN', 'DCNv2', 'DLRM'])
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--train-csv-file',default='/home/twsugkm569/hailey/DeepLight/data/large/train.csv') # train dataset
    parser.add_argument('--valid-csv-file',default='/home/twsugkm569/hailey/DeepLight/data/large/valid.csv') # valid dataset
    parser.add_argument('--feature-map-file',default='/home/twsugkm569/kevin/criteo_feature_map')
    parser.add_argument('--feature-map-from-file',action='store_true',default=False)
    parser.add_argument('--sparse_dim_start', type=int, default=13)
    parser.add_argument('--pep-flag', action='store_true', default=False)
    parser.add_argument("--threshold_type", type = str, default = 'dimension')
    parser.add_argument("--threshold_init", type = int, default = -150)
    parser.add_argument("--emb_save_path", default='./tmp/embedding/{task}')
    parser.add_argument("--retrain", action='store_true', default=False)
    parser.add_argument("--retrain_emb_sparsity", type=float, default=0)
    parser.add_argument("--re_init", action='store_true', default=False)
    
    global pars
    pars = parser.parse_args()
    print(pars)
    np.random.seed(pars.random_seed) 
    random.seed(pars.random_seed)
    torch.manual_seed(pars.random_seed)
    torch.cuda.manual_seed(pars.random_seed)

    if pars.save_path is None : 
        pars.save_path = './runs/'+str(pars.model)+"_IP8_"+str(pars.embedding_size)+"_FC"+pars.fc_sparsity
        if pars.model == 'DLRM':
            pars.save_path += "_"+pars.interaction_op
            if pars.interaction_op=="mix":
                pars.save_path += str(pars.compressed_dim)+"_expansion_factor"+str(pars.expansion_factor)+"_mix_act"+pars.mix_act

        if pars.lr_decay != "none":
            pars.save_path += "_lr_decay_"+pars.lr_decay

    if not os.path.exists(pars.save_path):
        os.makedirs(pars.save_path)

    global _writer
    _writer = SummaryWriter(pars.save_path)

    criteo_num_feat_dim = [i for i in range(1, pars.sparse_dim_start+1)]
    print('Loading training data...')
    if pars.debug:
        pars.n_epochs = 1
        result_dict = data_preprocess.read_data('/home/twsugkm569/Gary/org_DeepLight/tiny_train_input.csv', '/home/twsugkm569/kevin/criteo_feature_map', criteo_num_feat_dim, feature_dim_start=1, dim=39)
        test_dict = data_preprocess.read_data('/home/twsugkm569/Gary/org_DeepLight/tiny_test_input.csv', '/home/twsugkm569/kevin/criteo_feature_map', criteo_num_feat_dim, feature_dim_start=1, dim=39)
    else:        
        #result_dict = data_preprocess.read_data('./data/large/train.csv', './data/large/criteo_feature_map', criteo_num_feat_dim, feature_dim_start=1, dim=39)
        #test_dict = data_preprocess.read_data('./data/large/valid.csv', './data/large/criteo_feature_map', criteo_num_feat_dim, feature_dim_start=1, dim=39)
        data_start = time()
        result_dict = data_preprocess.read_data( pars.train_csv_file, pars.feature_map_file , criteo_num_feat_dim, feature_dim_start=1, dim=39, sparse_dim_start=pars.sparse_dim_start)
        test_dict = data_preprocess.read_data( pars.valid_csv_file, pars.feature_map_file , criteo_num_feat_dim, feature_dim_start=1, dim=39, sparse_dim_start=pars.sparse_dim_start)
        data_end = time()
        print("data loading time: ", data_end - data_start)
        if pars.feature_map_from_file:
            with open(pars.feature_map_file+'_count','r') as feature_map_file_:
                result_dict['feature_sizes'] = np.fromstring(feature_map_file_.readline(), dtype=int, sep=",")
        

    # global use_cuda, device
    use_cuda = pars.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    opt = {
        'device': device,
        'latent_dim': pars.embedding_size,
        'threshold_type': pars.threshold_type,
        'threshold_init': pars.threshold_init,
        'emb_save_path': pars.emb_save_path + '/sparsity_{sparsity}_total_params_{total_params}.txt', #+ '/{overall_sparsity}/{embid}.npy',
        'retrain_emb_sparsity': pars.retrain_emb_sparsity,
        're_init': pars.re_init
    }
    global model
    if pars.model == 'FM':
        model = FM(
            field_size=39, feature_sizes=result_dict['feature_sizes'], numerical=pars.numerical,
            embedding_size=pars.embedding_size, fc_sparsity=pars.fc_sparsity, pep_flag=pars.pep_flag, opt=opt,
        )
    elif pars.model == 'DeepFM':
        model = DeepFM(
            field_size=39, feature_sizes=result_dict['feature_sizes'], numerical=pars.numerical,
            embedding_size=pars.embedding_size, fc_sparsity=pars.fc_sparsity, pep_flag=pars.pep_flag, opt=opt,
        )
    elif pars.model == 'xDeepFM':
        model = xDeepFM(
            field_size=39, feature_sizes=result_dict['feature_sizes'], numerical=pars.numerical,
            embedding_size=pars.embedding_size, fc_sparsity=pars.fc_sparsity, pep_flag=pars.pep_flag, opt=opt,
        )
    elif pars.model == 'DeepFwFM':
        model = DeepFwFM(
            field_size=39, feature_sizes=result_dict['feature_sizes'], numerical=pars.numerical, device=device,
            embedding_size=pars.embedding_size, fc_sparsity=pars.fc_sparsity, pep_flag=pars.pep_flag, opt=opt,
        )
    elif pars.model == 'AutoInt':
        model = AutoInt(
            field_size=39, feature_sizes=result_dict['feature_sizes'], numerical=pars.numerical, device=device,
            embedding_size=pars.embedding_size, fc_sparsity=pars.fc_sparsity, pep_flag=pars.pep_flag, opt=opt,
        )
    elif pars.model == 'DCN':
        model = DCN(
            field_size=39, feature_sizes=result_dict['feature_sizes'], numerical=pars.numerical, device=device,
            embedding_size=pars.embedding_size, fc_sparsity=pars.fc_sparsity, use_DCNv2 = False, pep_flag=pars.pep_flag, opt=opt,
        )
    elif pars.model == 'DCNv2':
        model = DCN(
            field_size=39, feature_sizes=result_dict['feature_sizes'], numerical=pars.numerical,
            embedding_size=pars.embedding_size, fc_sparsity=pars.fc_sparsity, use_DCNv2 = True, pep_flag=pars.pep_flag, opt=opt,
        )
    elif pars.model == 'DLRM':
        model = DLRM(
            eval_only=pars.eval_only, field_size=39, feature_sizes=result_dict['feature_sizes'], numerical=pars.numerical,
            embedding_size=pars.embedding_size, fc_sparsity=pars.fc_sparsity, bottom_mlp=pars.bottom_mlp,top_mlp=pars.top_mlp,
            interaction_op=pars.interaction_op, full_interactions=pars.full_interactions,
            compressed_dim=pars.compressed_dim, expansion_factor=pars.expansion_factor, mix_act=pars.mix_act,
            mix_residual=pars.mix_residual, device=device, test_batch_size=pars.test_batch_size,
        )
    else:
        sys.exit("ERROR: --model={} is not supported.".format(pars.model))
    
    print(model)
    
    global candidate_p
    candidate_p = [int(sum(model.feature_sizes)*model.embedding_size * factor) for factor in [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.125, 0.1, 0.05, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.0125, 0.01]]


    if not pars.eval_only:
        model = model.to(device)
        skip_upto_epoch = 0
        skip_upto_batch = 0
        Best_auc = 0
        Best_acc = 0
        Best_model = False
        is_valid = False
        Xi_train = np.array(result_dict['index']).reshape((-1, model.field_size-pars.sparse_dim_start, 1))
        Xv_train = np.array(result_dict['value'])
        y_train = np.array(result_dict['label'])
        x_size = Xi_train.shape[0]
        print('Train split size:',x_size)

        Xi_valid = np.array(test_dict['index']).reshape((-1, model.field_size-pars.sparse_dim_start, 1))
        Xv_valid = np.array(test_dict['value'])
        y_valid = np.array(test_dict['label'])
        x_valid_size = Xi_valid.shape[0]
        is_valid = True
        
        """
            Training 
        """
        model = model.train()
        if pars.optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=pars.learning_rate, momentum=pars.momentum, weight_decay=pars.weight_decay)
        elif pars.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=pars.learning_rate, weight_decay=pars.weight_decay)
        elif pars.optimizer_type == 'rmsp':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=pars.learning_rate, weight_decay=pars.weight_decay)
        elif pars.optimizer_type == 'adag':
            optimizer = torch.optim.Adagrad(model.parameters(), lr=pars.learning_rate, weight_decay=pars.weight_decay)
        criterion = F.binary_cross_entropy_with_logits
        
        if pars.pretrain is not None:
            print('load pretrained weights', pars.pretrain)
            ld_model = torch.load(pars.pretrain)
            model.load_state_dict(ld_model["state_dict"])
            optimizer.load_state_dict(ld_model["opt_state_dict"])
            skip_upto_epoch = ld_model["epoch"]  # epochs
            skip_upto_batch = ld_model["iter"]  # batches
        else:
            print('init_weights')
            init_weights(model)
        
        train_result = []
        valid_result = []

        for epoch in range(pars.n_epochs):
            batch_iter = x_size // pars.batch_size
            if epoch < skip_upto_epoch:
                epoch += 1
                continue

            epoch_begin_time = time()
            for i in range(batch_iter+1):
                if epoch <= skip_upto_epoch and i < skip_upto_batch:
                    continue

                total_loss = 0.0
                offset = i*pars.batch_size
                end = min(x_size, offset+pars.batch_size)
                if offset == end:
                    break
                batch_xi = Variable(torch.LongTensor(Xi_train[offset:end]))
                batch_xv = Variable(torch.FloatTensor(Xv_train[offset:end]))
                batch_y = Variable(torch.FloatTensor(y_train[offset:end]))
                if use_cuda:
                    batch_xi, batch_xv, batch_y = batch_xi.cuda(), batch_xv.cuda(), batch_y.cuda()
                optimizer.zero_grad()
                outputs = model(batch_xi, batch_xv)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                adjust_learning_rate(optimizer, epoch=epoch, batch_size=pars.batch_size, init_lr=pars.learning_rate, \
                    n_epochs=pars.n_epochs, warmup=pars.warm, alpha=pars.lr_alpha, schedule=pars.lr_decay)

                total_loss += loss.data.item()
                log_iter = batch_iter * epoch + i + 1
                _writer.add_scalar("Train/Loss", total_loss, log_iter)
                _writer.add_scalar("Train/Learning rate", learning_rate, log_iter)

                if pars.pep_flag:
                    sparsity_l, params_l, overall_sparsity, total_params = calc_sparsity(model)
                    if not pars.retrain:
                        save_pruned_embedding(model, sparsity_l, params_l, overall_sparsity, total_params)
                    _writer.add_scalar('train/step_wise/mf_loss', loss.item(), log_iter)
                    _writer.add_scalar('train/step_wise/emb_sparsity', overall_sparsity, log_iter)
                    if i % 2000 == 0:
                        print('[Epoch {}|Step {}|Sparsity {:.4f}|Params {}]'.format(epoch+1, log_iter+1, overall_sparsity, total_params))
                

                
            if pars.pep_flag:  
                thr_list = get_threshold(model)
                for i in range(len(thr_list)):
                    _writer.add_histogram('threshold/'+str(i), thr_list[i], epoch)
                sparsity_l, params_l, overall_sparsity, total_params = calc_sparsity(model)
                _writer.add_scalar('train/epoch_wise/emb_sparsity', overall_sparsity, epoch)
                _writer.add_scalar('train/epoch_wise/params', total_params, epoch)
        

            torch.cuda.synchronize()
            epoch_end_time = time()

            train_loss, train_eval, _, _, train_acc, _= eval_by_batch(pars.eval_only, model, Xi_train, Xv_train, y_train, x_size, pars.test_batch_size)
            train_result.append(train_eval)
            print('Training [%d] loss: %.6f AUC: %.6f Acc: %.6f  time: %.3f s' %
                  (epoch + 1, train_loss, train_eval, train_acc, epoch_end_time-epoch_begin_time))
            if is_valid:
                valid_loss, valid_eval, TPR, TNR, valid_acc, inf_time= eval_by_batch(pars.eval_only, model, Xi_valid, Xv_valid, y_valid, x_valid_size, pars.test_batch_size)
                valid_result.append(valid_eval)
                print('Validation [%d] loss: %.6f AUC: %.6f TPR: %.6f TNR: %.6f Acc: %.6f Kinf/s: %.3f' %
                      (epoch + 1, valid_loss, valid_eval, TPR, TNR, valid_acc, inf_time))
                _writer.add_scalar("Test/AUC", valid_eval, epoch+1)
                _writer.add_scalar("Test/Acc", valid_acc, epoch+1)
                _writer.add_scalar("Test/TPR", TPR, epoch+1)
                _writer.add_scalar("Test/TNR", TNR, epoch+1)
                
                if valid_eval > Best_auc :
                    Best_model = True
                    Best_auc = valid_eval
                if valid_acc > Best_acc :
                    Best_acc = valid_acc
                    print('Best ACC: %.6f' % (Best_acc))

                if Best_model:
                    print('Best AUC: %.6f ACC: %.6f' % (Best_auc , valid_acc))
            print('*' * 50)
            
            permute_idx = np.random.permutation(x_size)
            Xi_train = Xi_train[permute_idx]
            Xv_train = Xv_train[permute_idx]
            y_train = y_train[permute_idx]
            print('Training dataset shuffled.')

            ckpt = {
                'nepochs': pars.n_epochs,
                'nbatches': batch_iter,
                'epoch': epoch,
                'iter': i + 1,
                'state_dict': model.state_dict(),
                'opt_state_dict': optimizer.state_dict()
            }
            if pars.save_path and Best_model:
                print('Saving best model at ', pars.save_path)
                torch.save(ckpt, pars.save_path+"/best.pt")
                Best_model = False

            if pars.save_path and epoch==(pars.warm-1):
                torch.save(ckpt, pars.save_path+"/epoch{}.pt".format(pars.warm))

        """
            Post Evaluation 
        """
        post_eval(pars.eval_only, model, Xi_valid, Xv_valid, y_valid, load_path=pars.save_path+"/best.pt", test_batch_size=pars.test_batch_size)

    else:
        model = model.to(device)
        Xi_valid = np.array(test_dict['index']).reshape((-1, model.field_size-pars.sparse_dim_start, 1))
        Xv_valid = np.array(test_dict['value'])
        y_valid = np.array(test_dict['label'])
        post_eval(pars.eval_only, model, Xi_valid, Xv_valid, y_valid, load_path=pars.load_path+"/best.pt", test_batch_size=pars.test_batch_size)
