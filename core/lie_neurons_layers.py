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

from core.lie_alg_util import *

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
    def __init__(self, in_channels, algebra_type='sl3', share_nonlinearity=False, leaky_relu=False, negative_slope=0.2):
        super(LNKillingRelu, self).__init__()
        self.share_nonlinearity = share_nonlinearity

        if share_nonlinearity == True:
            self.learn_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.learn_dir = nn.Linear(in_channels, in_channels, bias=False)

        self.HatLayer = HatLayer(algebra_type)
        self.algebra_type = algebra_type
        self.leaky_relu = leaky_relu
        self.negative_slope = negative_slope

    def forward(self, x):
        '''
        x: point features of shape [B, F, 8, N]
        '''
        # B, F, _, N = x.shape
        x_out = torch.zeros_like(x)

        d = self.learn_dir(x.transpose(1, -1)).transpose(1, -1)

        x = x.transpose(2, -1)
        d = d.transpose(2, -1)

        x_hat = self.HatLayer(x)
        d_hat = self.HatLayer(d)
        kf_xd = killingform(x_hat, d_hat, self.algebra_type)
        
        if self.leaky_relu:
            mask = (kf_xd <= 0).float()
            x_out = self.negative_slope * x + (1-self.negative_slope ) \
                *(mask*x + (1-mask)*(x-(-kf_xd)*d))
        else:
            x_out = torch.where(kf_xd <= 0, x, x - (-kf_xd) * d)
        x_out = x_out.transpose(2, -1)

        return x_out


class LNLieBracket(nn.Module):
    def __init__(self, in_channels, algebra_type='sl3', share_nonlinearity=False):
        super(LNLieBracket, self).__init__()
        self.share_nonlinearity = share_nonlinearity
        self.algebra_type = algebra_type

        if share_nonlinearity == True:
            self.learn_dir = nn.Linear(in_channels, 1, bias=False)
            self.learn_dir2 = nn.Linear(in_channels, 1, bias=False)
        else:
            self.learn_dir = nn.Linear(in_channels, in_channels, bias=False)
            self.learn_dir2 = nn.Linear(in_channels, in_channels, bias=False)

        self.HatLayer =  HatLayer(algebra_type)


    def forward(self, x):
        '''
        x: point features of shape [B, F, K, N]
        '''
        # B, F, _, N = x.shape

        d = self.learn_dir(x.transpose(1, -1)).transpose(1, -1)         # [B, F, K, N]
        d2 = self.learn_dir2(x.transpose(1, -1)).transpose(1, -1)       # [B, F, K, N]
        d = d.transpose(2, -1)

        d2 = d2.transpose(2,-1)

        d_hat = self.HatLayer(d)
        d2_hat = self.HatLayer(d2)
        lie_bracket = torch.matmul(d2_hat, d_hat) - torch.matmul(d_hat,d2_hat)
        x_out = x + vee(lie_bracket,self.algebra_type).transpose(2, -1)
        return x_out
    

class LNLieBracketNoResidualConnect(nn.Module):
    def __init__(self, in_channels, algebra_type='sl3', share_nonlinearity=False):
        super(LNLieBracketNoResidualConnect, self).__init__()
        self.share_nonlinearity = share_nonlinearity
        self.algebra_type = algebra_type

        if share_nonlinearity == True:
            self.learn_dir = nn.Linear(in_channels, 1, bias=False)
            self.learn_dir2 = nn.Linear(in_channels, 1, bias=False)
        else:
            self.learn_dir = nn.Linear(in_channels, in_channels, bias=False)
            self.learn_dir2 = nn.Linear(in_channels, in_channels, bias=False)

        # torch.nn.init.uniform_(self.learn_dir.weight, a=0.0, b=0.5)
        # torch.nn.init.uniform_(self.learn_dir2.weight, a=0.0, b=0.5)

        self.HatLayer = HatLayer()
        self.relu = LNKillingRelu(
            in_channels, share_nonlinearity=share_nonlinearity)


    def forward(self, x):
        '''
        x: point features of shape [B, F, 8, N]
        '''
        # B, F, _, N = x.shape

        d = self.learn_dir(x.transpose(1, -1)).transpose(1, -1)
        d2 = self.learn_dir2(x.transpose(1, -1)).transpose(1, -1)

        d = d.transpose(2, -1)
        d2 = d2.transpose(2,-1)

        d_hat = self.HatLayer(d)
        d2_hat = self.HatLayer(d2)
        lie_bracket = torch.matmul(d2_hat, d_hat) - torch.matmul(d_hat,d2_hat)
        x_out = vee(lie_bracket,self.algebra_type).transpose(2, -1)
        
        # print("avg XY: ", torch.mean(torch.matmul(d2_hat, d_hat)))
        # print("avg YX: ", torch.mean(torch.matmul(d_hat,d2_hat)))
        # print("avg out: ", torch.mean(x_out))

        # print("median XY: ", torch.median(torch.matmul(d2_hat, d_hat)))
        # print("median YX: ", torch.median(torch.matmul(d_hat,d2_hat)))
        # print("median out: ", torch.median(x_out))

        # print("max XY: ", torch.max(torch.matmul(d2_hat, d_hat)))
        # print("max YX: ", torch.max(torch.matmul(d_hat,d2_hat)))
        # print("max out: ", torch.max(x_out))

        # print("min XY: ", torch.min(torch.abs(torch.matmul(d2_hat, d_hat))))
        # print("min YX: ", torch.min(torch.abs(torch.matmul(d_hat,d2_hat))))
        # print("min out: ", torch.min(torch.abs(x_out)))

        # print("avg X: ", torch.mean(x))
        # print("median X: ", torch.median(x))
        # print("max X: ", torch.max(x))
        # print("min X: ", torch.min(torch.abs(x)))
        # print("--------------------------------------------")

        return x_out


class LNEquivairanChannelMixing(nn.Module):
    def __init__(self, in_channel) -> None:
        super(LNEquivairanChannelMixing).__init__()

        self.ln_linear_relu = LNLinearAndKillingRelu(in_channel, 3, algebra_type='so3', share_nonlinearity=False),
        self.ln_pooling = LNMaxPool(3)

    def forward(self, x):
        x = self.ln_linear_relu(x)  # B, F, K (3 for so(3)), N
        x = self.ln_pooling(x).squeeze(-1)      # B, F, K (3 for so(3))
        out = torch.matmul(x.transpose(1, 2), x)
        return out
    
class LNLieBracketChannelMix(nn.Module):
    def __init__(self, in_channels, algebra_type='so3', share_nonlinearity=False, residual_connect=True):
        super(LNLieBracketChannelMix, self).__init__()
        self.share_nonlinearity = share_nonlinearity
        self.algebra_type = algebra_type
        self.residual_connect = residual_connect

        if share_nonlinearity == True:
            self.learn_dir = nn.Linear(in_channels, 1, bias=False)
            self.learn_dir2 = nn.Linear(in_channels, 1, bias=False)
        else:
            self.learn_dir = nn.Linear(in_channels, in_channels, bias=False)
            self.learn_dir2 = nn.Linear(in_channels, in_channels, bias=False)

        self.HatLayer =  HatLayer(algebra_type)


    def forward(self, x, M1=torch.eye(3), M2=torch.eye(3)):
        '''
        x: point features of shape [B, F, K, N]
        '''
        # B, F, _, N = x.shape
        M1 = M1.to(x.device)
        M2 = M2.to(x.device)

        d = self.learn_dir(x.transpose(1, -1)).transpose(1, -1)     # [B, F, K, N]
        d2 = self.learn_dir2(x.transpose(1, -1)).transpose(1, -1)   # [B, F, K, N]

        d = torch.einsum('d k, b f k n -> b f n d',M1,d)             # [B, F, N, K]
        d2 = torch.einsum('d k, b f k n -> b f n d',M2,d2)           # [B, F, N, K]                      

        d_hat = self.HatLayer(d)
        d2_hat = self.HatLayer(d2)
        lie_bracket = torch.matmul(d2_hat, d_hat) - torch.matmul(d_hat,d2_hat)
        if self.residual_connect:
            x_out = x + vee(lie_bracket,self.algebra_type).transpose(2, -1)
        else:
            x_out = x + vee(lie_bracket,self.algebra_type).transpose(2, -1)
        return x_out


class LNLinearAndKillingRelu(nn.Module):
    def __init__(self, in_channels, out_channels, algebra_type='sl3', share_nonlinearity=False, leaky_relu=False,negative_slope=0.2):
        super(LNLinearAndKillingRelu, self).__init__()
        self.share_nonlinearity = share_nonlinearity

        self.linear = LNLinear(in_channels, out_channels)
        self.leaky_relu = LNKillingRelu(
            out_channels, algebra_type=algebra_type, share_nonlinearity=share_nonlinearity, leaky_relu=leaky_relu, negative_slope=negative_slope)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        x = self.linear(x)
        # LeakyReLU
        x_out = self.leaky_relu(x)

        return x_out

class LNLinearAndLieBracket(nn.Module):
    def __init__(self, in_channels, out_channels, algebra_type='sl3', share_nonlinearity=False):
        super(LNLinearAndLieBracket, self).__init__()
        self.share_nonlinearity = share_nonlinearity

        self.linear = LNLinear(in_channels, out_channels)
        self.liebracket = LNLieBracket(
            out_channels, algebra_type=algebra_type, share_nonlinearity=share_nonlinearity)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        x = self.linear(x)
        # Bracket
        x_out = self.liebracket(x)

        return x_out
    

class LNLinearAndLieBracketChannelMix(nn.Module):
    def __init__(self, in_channels, out_channels, algebra_type='so3', share_nonlinearity=False, residual_connect=True):
        super(LNLinearAndLieBracketChannelMix, self).__init__()
        self.share_nonlinearity = share_nonlinearity
        self.linear = LNLinear(in_channels, out_channels)
        self.liebracket = LNLieBracketChannelMix(
            out_channels, algebra_type=algebra_type, share_nonlinearity=share_nonlinearity, residual_connect=residual_connect)

    def forward(self, x, M1, M2):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        x = self.linear(x)
        # Bracket
        x_out = self.liebracket(x, M1, M2)

        return x_out

class LNLinearAndLieBracketNoResidualConnect(nn.Module):
    def __init__(self, in_channels, out_channels, algebra_type='sl3', share_nonlinearity=False):
        super(LNLinearAndLieBracketNoResidualConnect, self).__init__()
        self.share_nonlinearity = share_nonlinearity

        self.linear = LNLinear(in_channels, out_channels)
        self.liebracket = LNLieBracketNoResidualConnect(
            out_channels, algebra_type=algebra_type, share_nonlinearity=share_nonlinearity)
        
        # torch.nn.init.uniform_(self.linear.fc.weight, a=0.0, b=0.5)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        x = self.linear(x)
        # Bracket
        x_out = self.liebracket(x)

        return x_out

class LNMaxPool(nn.Module):
    def __init__(self, in_channels, algebra_type='sl3',abs_killing_form=False):
        super(LNMaxPool, self).__init__()
        self.learn_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.absolute = abs_killing_form
        self.algebra_type = algebra_type
        self.hat_layer = HatLayer(algebra_type=algebra_type)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        B, F, K, N = x.shape

        d = self.learn_dir(x.transpose(1, -1)).transpose(1, -1)

        x_hat = self.hat_layer(x.transpose(2, -1))
        d_hat = self.hat_layer(d.transpose(2, -1))
        killing_forms = killingform(x_hat, d_hat, self.algebra_type).squeeze(-1)


        if not self.absolute:
            idx = killing_forms.max(dim=-1, keepdim=False)[1]
        else:
            idx = killing_forms.abs().max(dim=-1, keepdim=False)[1]

        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]], indexing="ij")\
            + (idx.reshape(B, F, 1).repeat(1, 1, K),)
        x_max = x[index_tuple]
        return x_max


class LNLinearAndKillingReluAndPooling(nn.Module):
    def __init__(self, in_channels, out_channels, algebra_type='sl3', share_nonlinearity=False, abs_killing_form=False,
                 use_batch_norm=False, dim=5):
        super(LNLinearAndKillingReluAndPooling, self).__init__()
        self.share_nonlinearity = share_nonlinearity

        self.linear = LNLinear(in_channels, out_channels)
        self.leaky_relu = LNKillingRelu(
            out_channels, algebra_type=algebra_type, share_nonlinearity=share_nonlinearity)
        self.max_pool = LNMaxPool(
            out_channels, algebra_type=algebra_type, abs_killing_form=abs_killing_form)
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
    def __init__(self, num_features, dim, algebra_type='sl3', affine=False, momentum=0.1):
        super(LNBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features, affine=affine, momentum=momentum)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features, affine=affine, momentum=momentum)

        self.hat_layer = HatLayer(algebra_type=algebra_type)
        self.algebra_type = algebra_type

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, K, N_samples, ...]
        '''

        x_hat = self.hat_layer(x.transpose(2, -1))
        kf = killingform(x_hat, x_hat,algebra_type=self.algebra_type)
        # b, f, n, _, _ = x_hat.shape
        # kf = rearrange(torch.det(
        #         rearrange(x_hat, 'b f n m1 m2 -> (b f n) m1 m2')), '(b f n) -> b f n 1', b=b, f=f, n=n)
        kf = torch.where(kf <= 0, torch.clamp(
            kf, max=-EPS), torch.clamp(kf, min=EPS))

        # kf = torch.clamp(torch.abs(kf), min=EPS)
        kf = kf.squeeze(-1)

        # kf = compute_killing_form(x, x) + EPS
        
        kf_bn = self.bn(kf)
        # kf_bn = torch.clamp(torch.abs(self.bn(kf)), min=EPS)
        
        kf = kf.unsqueeze(2)
        kf_bn = kf_bn.unsqueeze(2)
        x = x / kf * kf_bn

        return x


class LNInvariant(nn.Module):
    def __init__(self, in_channel, algebra_type='sl3', dir_dim=8, method='learned_killing'):
        super(LNInvariant, self).__init__()

        self.hat_layer = HatLayer(algebra_type=algebra_type)
        self.learned_dir = LNLinearAndKillingRelu(
            in_channel, dir_dim, share_nonlinearity=True)
        self.method = method
        self.algebra_type = algebra_type

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, K, N_samples, ...]
        '''
        x_hat = self.hat_layer(x.transpose(2, -1))
        if self.method == 'learned_killing':
            d_hat = self.hat_layer(self.learned_dir(x).transpose(2, -1))
            x_out = killingform(x_hat, d_hat, algebra_type=self.algebra_type,feature_wise=True)
        elif self.method == 'self_killing':
            x_out = killingform(x_hat, x_hat,algebra_type=self.algebra_type)
        elif self.method == 'det':
            b, f, n, _, _ = x_hat.shape
            x_out = rearrange(torch.det(
                rearrange(x_hat, 'b f n m1 m2 -> (b f n) m1 m2')), '(b f n) -> b f n 1', b=b, f=f, n=n)
        elif self.method == 'trace':
            x_out = (x_hat.transpose(-1, -2) *
                     x_hat).sum(dim=(-1, -2))[..., None]

        return x_out
