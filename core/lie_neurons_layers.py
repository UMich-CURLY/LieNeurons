from core.lie_group_util import *
import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('.')


EPS = 1e-6


class LNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LNLinear, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        x: input of shape [B, F, 8, N]
        '''
        x_out = self.fc(x.transpose(1, -1)).transpose(1, -1)
        return x_out


class LNKillingRelu(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False):
        super(LNKillingRelu, self).__init__()
        self.share_nonlinearity = share_nonlinearity
        if share_nonlinearity == True:
            self.learn_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.learn_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, F, 8, N]
        '''
        B, F, _, N = x.shape
        x_out = torch.zeros_like(x)

        d = self.learn_dir(x.transpose(1, -1)).transpose(1, -1)
        for b in range(B):
            for f in range(F):
                for n in range(N):
                    # generate elements in sl3
                    X = R8_to_sl3(x[b, f, :, n])
                    if not self.share_nonlinearity:
                        cur_d = d[b, f, :, n]
                    else:
                        cur_d = d[b, 0, :, n]

                    D = R8_to_sl3(cur_d)

                    # TODO: move killing form computation to a function and support multiple groups
                    killing_form = 6*torch.trace(X@D)
                    if killing_form < 0:
                        x_out[b, f, :, n] = x[b, f, :, n]
                    else:
                        x_out[b, f, :, n] = x[b, f, :, n] - \
                            (-killing_form) * cur_d

        return x_out


class LNLinearAndKillingNonLinear(nn.Module):
    def __init__(self, in_channels, out_channels, share_nonlinearity=False):
        super(LNLinearAndKillingNonLinear, self).__init__()
        self.share_nonlinearity = share_nonlinearity

        self.linear = LNLinear(in_channels, out_channels)
        self.leaky_relu = LNKillingRelu(
            out_channels, share_nonlinearity=share_nonlinearity)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Conv
        x = self.linear(x)
        # LeakyReLU
        x_out = self.leaky_relu(x)

        return x_out


class LNMaxPool(nn.Module):
    def __init__(self, in_channels, abs_killing_form=False):
        super(LNMaxPool, self).__init__()
        self.learn_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.absolute = abs_killing_form

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        B, F, K, N = x.shape
        killing_forms = torch.zeros([B, F, N])

        d = self.learn_dir(x.transpose(1, -1)).transpose(1, -1)

        killing_forms = compute_killing_form(x, d)

        if not self.absolute:
            idx = killing_forms.max(dim=-1, keepdim=False)[1]
        else:
            idx = killing_forms.abs().max(dim=-1, keepdim=False)[1]

        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]], indexing="ij")\
            + (idx.reshape(B, F, 1).repeat(1, 1, K),)
        x_max = x[index_tuple]
        return x_max


class LNLinearAndKillingNonLinearAndPooling(nn.Module):
    def __init__(self, in_channels, out_channels, share_nonlinearity=False, abs_killing_form=False,\
                  use_batch_norm=False, dim=5):
        super(LNLinearAndKillingNonLinearAndPooling, self).__init__()
        self.share_nonlinearity = share_nonlinearity

        self.linear = LNLinear(in_channels, out_channels)
        self.leaky_relu = LNKillingRelu(
            out_channels, share_nonlinearity=share_nonlinearity)
        self.max_pool = LNMaxPool(out_channels, abs_killing_form=abs_killing_form)
        self.use_batch_norm = use_batch_norm

        if use_batch_norm:
            self.batch_norm = LNBatchNorm(out_channels, dim=dim)
            

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        x = self.linear(x)

        if self.use_batch_norm:
            x = self.batch_norm(x)

        # LeakyReLU
        x_out = self.leaky_relu(x)

        x_out = self.max_pool(x_out)
        return x_out


class LNBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(LNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''

        killing_forms = compute_killing_form(x, x) + EPS

        kf_bn = self.bn(killing_forms)
        killing_forms = killing_forms.unsqueeze(2)
        kf_bn = kf_bn.unsqueeze(2)
        x = x / killing_forms * kf_bn

        return x
