# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PEPMLP(nn.Module):
    # use PEP to replace torch.nn.Embedding
    def __init__(self, opt, layer, in_dim, out_dim):
        super(PEPMLP, self).__init__()
        self.layer = layer
        self.use_cuda = opt['use_cuda']
        self.threshold_type = opt['threshold_type']  # define the granularity of s
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.gk = 1
        init = opt['threshold_init']  # initial value of s
        self.retrain = False
        self.mask = None

        self.g = torch.sigmoid  # define the threshold function g
        self.s = self.init_threshold(init)
        # self.offsets = np.array((0, *np.cumsum(self.field_dims)[:-1]), dtype=np.long)
        # initialize the embedding matrix V
        self.v = torch.nn.Parameter(torch.rand(self.in_dim, self.out_dim))
        torch.nn.init.xavier_uniform_(self.v)

        if opt['retrain']:
            self.retrain = True
            self.init_retrain(opt)
            print("Retrain epoch {}".format(opt['retrain_mlp_param']))

        self.sparse_v = self.v.data

    def init_retrain(self, opt):
        retrain_mlp_param = opt['retrain_mlp_param']
        sparse_mlp = np.load(opt['mlp_save_path'].format(num_parameter=retrain_mlp_param, layer=self.layer)+'.npy')
        sparse_mlp = torch.from_numpy(sparse_mlp)
        mask = torch.abs(torch.sign(sparse_mlp))
        if opt['re_init']:
            init_mlp = torch.nn.Parameter(torch.rand(self.in_dim, self.out_dim))
            torch.nn.init.xavier_uniform_(init_mlp)
        else:
            init_mlp = np.load(opt['mlp_save_path'].format(num_parameter='initial_mlp', emb=self.embidx) + '.npy')
            init_mlp = torch.from_numpy(init_mlp)

        init_mlp = init_mlp * mask
        self.v = torch.nn.Parameter(init_mlp)
        self.mask = mask
        self.gk = 0
        if self.use_cuda:
            self.mask = self.mask.cuda()

    def init_threshold(self, init):  # initialize threshold with different granularities
        if self.threshold_type == 'global':
            s = nn.Parameter(init * torch.ones(1)) # 即 field 因為26個emb分開建
        elif self.threshold_type == 'in_dim':
            s = nn.Parameter(init * torch.ones([self.in_dim]))
        elif self.threshold_type == 'out_dim':
            s = nn.Parameter(init * torch.ones([self.out_dim]))
        # elif self.threshold_type == 'field':
            # s = nn.Parameter(init * torch.ones([self.field_num, 1]))
        elif self.threshold_type == 'feature_dim':
            s = nn.Parameter(init * torch.ones([self.in_dim, self.out_dim]))
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
        xv = F.linear(x, self.sparse_v)

        return xv


