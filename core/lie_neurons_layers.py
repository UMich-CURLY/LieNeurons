from core.lie_group_util import *
from core.lie_alg_util import *
import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

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

        self.HatLayerSl3 = HatLayerSl3()

    def forward(self, x):
        '''
        x: point features of shape [B, F, 8, N]
        '''
        B, F, _, N = x.shape
        x_out = torch.zeros_like(x)

        d = self.learn_dir(x.transpose(1, -1)).transpose(1, -1)

        x = x.transpose(2, -1)
        d = d.transpose(2, -1)

        x_hat = self.HatLayerSl3(x)
        d_hat = self.HatLayerSl3(d)
        killing_form = killingform_sl3(x_hat, d_hat)
        x_out = torch.where(killing_form < 0, x, x - (-killing_form) * d)
        x_out = x_out.transpose(2, -1)

        return x_out


class LNLinearAndKillingRelu(nn.Module):
    def __init__(self, in_channels, out_channels, share_nonlinearity=False):
        super(LNLinearAndKillingRelu, self).__init__()
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
        self.hat_layer = HatLayerSl3()

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        B, F, K, N = x.shape
        # killing_forms = torch.zeros([B, F, N])

        d = self.learn_dir(x.transpose(1, -1)).transpose(1, -1)

        x_hat = self.hat_layer(x.transpose(2, -1))
        d_hat = self.hat_layer(d.transpose(2, -1))
        killing_forms = killingform_sl3(x_hat, d_hat).squeeze(-1)

        # killing_forms = compute_killing_form(x, d)

        if not self.absolute:
            idx = killing_forms.max(dim=-1, keepdim=False)[1]
        else:
            idx = killing_forms.abs().max(dim=-1, keepdim=False)[1]

        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]], indexing="ij")\
            + (idx.reshape(B, F, 1).repeat(1, 1, K),)
        x_max = x[index_tuple]
        return x_max


class LNLinearAndKillingReluAndPooling(nn.Module):
    def __init__(self, in_channels, out_channels, share_nonlinearity=False, abs_killing_form=False,
                 use_batch_norm=False, dim=5):
        super(LNLinearAndKillingReluAndPooling, self).__init__()
        self.share_nonlinearity = share_nonlinearity

        self.linear = LNLinear(in_channels, out_channels)
        self.leaky_relu = LNKillingRelu(
            out_channels, share_nonlinearity=share_nonlinearity)
        self.max_pool = LNMaxPool(
            out_channels, abs_killing_form=abs_killing_form)
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

        self.hat_layer = HatLayerSl3()

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, K, N_samples, ...]
        '''

        x_hat = self.hat_layer(x.transpose(2, -1))
        killing_forms = killingform_sl3(x_hat, x_hat) + EPS
        killing_forms = killing_forms.squeeze(-1)

        # killing_forms = compute_killing_form(x, x) + EPS

        kf_bn = self.bn(killing_forms)
        killing_forms = killing_forms.unsqueeze(2)
        kf_bn = kf_bn.unsqueeze(2)
        x = x / killing_forms * kf_bn

        return x


class LNInvariantPooling(nn.Module):
    def __init__(self, method='killing'):
        super(LNInvariantPooling, self).__init__()

        self.hat_layer = HatLayerSl3()
        self.method = method

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, K, N_samples, ...]
        '''
        x_hat = self.hat_layer(x.transpose(2, -1))
        if self.method == 'killing':
            x_out = killingform_sl3(x_hat, x_hat)
        elif self.method == 'det':
            b, f, n, _, _ = x_hat.shape
            x_out = rearrange(torch.det(
                rearrange(x_hat, 'b f n m1 m2 -> (b f n) m1 m2')), '(b f n) -> b f n 1', b=b, f=f, n=n)
        elif self.method == 'trace':
            x_out = (x_hat.transpose(-1, -2) *
                     x_hat).sum(dim=(-1, -2))[..., None]

        return x_out
