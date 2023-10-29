# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PEPEmbedding(nn.Module):
    # use PEP to replace torch.nn.Embedding
    def __init__(self, opt, embid, idx_num):
        super(PEPEmbedding, self).__init__()
        self.embid = embid
        self.device = opt['device']
        self.threshold_type = opt['threshold_type']  # define the granularity of s
        self.latent_dim = opt['latent_dim']          # define the initial embedding size d
        # self.field_dims = opt['field_dims']
        self.idx_num = idx_num
        # self.idx_num = sum(self.field_dims)    # total numbers of features
        # self.field_num = len(self.field_dims)

        # self.g_type = opt['g_type']
        self.gk = 1
        init = opt['threshold_init']  # initial value of s
        self.retrain = False
        self.mask = None

        self.g = torch.sigmoid  # define the threshold function g
        self.s = self.init_threshold(init)
        # self.offsets = np.array((0, *np.cumsum(self.field_dims)[:-1]), dtype=np.long)
        # initialize the embedding matrix V
        self.v = torch.nn.Parameter(torch.rand(self.idx_num, self.latent_dim))
        torch.nn.init.xavier_uniform_(self.v)

        if opt['retrain_emb_sparsity'] != 0:
            self.retrain = True
            self.init_retrain(opt)
            print("Retrain sparsity {}".format(opt['retrain_emb_sparsity']))

        self.sparse_v = self.v.data

    def init_retrain(self, opt):
        retrain_emb_sparsity = opt['retrain_emb_sparsity']
        sparse_emb = np.load(opt['emb_save_path'].format(overall_sparsity=retrain_emb_sparsity, emb=self.embid) + '.npy')
        sparse_emb = torch.from_numpy(sparse_emb)
        mask = torch.abs(torch.sign(sparse_emb))
        if opt['re_init']:
            init_emb = torch.nn.Parameter(torch.rand(self.idx_num, self.latent_dim))
            torch.nn.init.xavier_uniform_(init_emb)
        else:
            init_emb = np.load(opt['emb_save_path'].format(overall_sparsity='initial_embedding', emb=self.embid) + '.npy')
            init_emb = torch.from_numpy(init_emb)

        init_emb = init_emb * mask
        self.v = torch.nn.Parameter(init_emb)
        self.mask = mask
        self.gk = 0
        self.mask = self.mask.to(self.device)

    def init_threshold(self, init):  # initialize threshold with different granularities
        if self.threshold_type == 'global':
            s = nn.Parameter(init * torch.ones(1)) # 即 field 因為26個emb分開建
        elif self.threshold_type == 'dimension':
            s = nn.Parameter(init * torch.ones([self.latent_dim]))
        elif self.threshold_type == 'index':
            s = nn.Parameter(init * torch.ones([self.idx_num, 1]))
        # elif self.threshold_type == 'field':
            # s = nn.Parameter(init * torch.ones([self.field_num, 1]))
        elif self.threshold_type == 'index_dim':
            s = nn.Parameter(init * torch.ones([self.idx_num, self.latent_dim]))
        # elif self.threshold_type == 'field_dim':
            # s = nn.Parameter(init * torch.ones([self.field_num, self.latent_dim]))
        else:
            raise ValueError('Invalid threshold_type: {}'.format(self.threshold_type))
        return s

    def soft_threshold(self, v, s):  # the reparameterization function S(V, D)
        # if s.size(0) == self.field_num:  # field-wise lambda
            # field_v = torch.split(v, tuple(self.field_dims))
            # concat_v = []
            # for i, v in enumerate(field_v):
                # v = torch.sign(v) * torch.relu(torch.abs(v) - (self.g(s[i]) * self.gk))
                # concat_v.append(v)

            # concat_v = torch.cat(concat_v, dim=0)
            # return concat_v
        # else:
        return torch.sign(v) * torch.relu(torch.abs(v) - (self.g(s) * self.gk))

    def forward(self, x):
        # x = x + x.new_tensor(self.offsets).unsqueeze(0)
        self.sparse_v = self.soft_threshold(self.v, self.s)  # the pruned sparse embedding matrix
        if self.retrain:
            self.sparse_v = self.sparse_v * self.mask
        # xv = F.embedding(x, self.sparse_v)  # retrieving feature embeddings by given feature x
        xv = F.embedding_bag(x, self.sparse_v, mode='sum')
        return xv


