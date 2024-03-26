import sys  # nopep8
sys.path.append('.')  # nopep8

import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


from core.lie_alg_util import *
from core.lie_neurons_layers import *
from core.vn_layers import *


class SO3EquivariantVNReluLayers(nn.Module):
    def __init__(self, in_channels):
        super(SO3EquivariantVNReluLayers, self).__init__()
        feat_dim = 1024
        share_nonlinearity = False
        leaky_relu = True
        self.ln_fc = VNLinearAndLeakyReLU(
            in_channels, feat_dim, share_nonlinearity=share_nonlinearity, leaky_relu=leaky_relu, dim=4)
        self.ln_fc2 = VNLinearAndLeakyReLU(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity,leaky_relu=leaky_relu, dim=4)
        self.ln_fc3 = VNLinearAndLeakyReLU(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity,leaky_relu=leaky_relu, dim=4)
        self.ln_fc4 = VNLinearAndLeakyReLU(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity,leaky_relu=leaky_relu, dim=4)

        self.fc_final = nn.Linear(feat_dim, 1, bias=False)

    def forward(self, x):
        '''
        x input of shape [B, F, 3, 1]
        '''
        x = self.ln_fc(x)
        x = self.ln_fc2(x)
        x = self.ln_fc3(x)
        x = self.ln_fc4(x)

        x = torch.permute(x, (0, 3, 2, 1))  # [B, 1, 3, F]
        x_out = rearrange(self.fc_final(x), 'b 1 k 1 -> b k')   # [B, 3]
        return x_out

class SO3EquivariantReluLayers(nn.Module):
    def __init__(self, in_channels):
        super(SO3EquivariantReluLayers, self).__init__()
        feat_dim = 1024
        share_nonlinearity = False
        leaky_relu = True
        self.ln_fc = LNLinearAndKillingRelu(
            in_channels, feat_dim, share_nonlinearity=share_nonlinearity, leaky_relu=leaky_relu, algebra_type='so3')
        self.ln_fc2 = LNLinearAndKillingRelu(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity,leaky_relu=leaky_relu, algebra_type='so3')
        self.ln_fc3 = LNLinearAndKillingRelu(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity,leaky_relu=leaky_relu, algebra_type='so3')
        self.ln_fc4 = LNLinearAndKillingRelu(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity,leaky_relu=leaky_relu, algebra_type='so3')
        
        self.fc_final = nn.Linear(feat_dim, 1, bias=False)

    def forward(self, x):
        '''
        x input of shape [B, F, 3, 1]
        '''
        x = self.ln_fc(x)   # [B, F, 3, 1]
        x = self.ln_fc2(x)  # [B, F, 3, 1]
        x = self.ln_fc3(x)
        x = self.ln_fc4(x)
        
        x = torch.permute(x, (0, 3, 2, 1))  # [B, 1, 3, F]
        x_out = rearrange(self.fc_final(x), 'b 1 k 1 -> b k')   # [B, 3]
        return x_out


class SO3EquivariantBracketLayers(nn.Module):
    def __init__(self, in_channels):
        super(SO3EquivariantBracketLayers, self).__init__()
        feat_dim = 1024
        share_nonlinearity = False
        self.ln_fc = LNLinearAndLieBracket(in_channels, feat_dim,share_nonlinearity=share_nonlinearity, algebra_type='so3')
        self.ln_fc2 = LNLinearAndLieBracket(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity, algebra_type='so3')
        self.ln_fc3 = LNLinearAndLieBracket(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity, algebra_type='so3')
        self.ln_fc4 = LNLinearAndLieBracket(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity, algebra_type='so3')
        # self.ln_fc5 = LNLinearAndLieBracket(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity, algebra_type='so3')
        # self.ln_fc6 = LNLinearAndLieBracket(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity, algebra_type='so3')
        # self.ln_fc7 = LNLinearAndLieBracket(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity, algebra_type='so3')
        
        self.fc_final = nn.Linear(feat_dim, 1, bias=False)

    def forward(self, x):
        '''
        x input of shape [B, F, 3, 1]
        '''
        x = self.ln_fc(x)   # [B, F, 3, 1]
        x = self.ln_fc2(x)  # [B, F, 3, 1]
        x = self.ln_fc3(x)
        x = self.ln_fc4(x)
        # x = self.ln_fc5(x)
        # x = self.ln_fc6(x)
        # x = self.ln_fc7(x)

        x = torch.permute(x, (0, 3, 2, 1))  # [B, 1, 3, F]
        x_out = rearrange(self.fc_final(x), 'b 1 k 1 -> b k')   # [B, 3]
        return x_out

class SO3EquivariantReluBracketLayers(nn.Module):
    def __init__(self, in_channels):
        super(SO3EquivariantReluBracketLayers, self).__init__()
        feat_dim = 1024
        share_nonlinearity = False
        leaky_relu = True
        self.ln_fc = LNLinearAndKillingRelu(
            feat_dim, feat_dim, share_nonlinearity=share_nonlinearity, leaky_relu=leaky_relu, algebra_type='so3')
        self.ln_fc2 = LNLinearAndKillingRelu(
            feat_dim, feat_dim, share_nonlinearity=share_nonlinearity, leaky_relu=leaky_relu, algebra_type='so3')
        self.ln_fc_bracket = LNLinearAndLieBracket(in_channels, feat_dim,share_nonlinearity=share_nonlinearity, algebra_type='so3')
        self.ln_fc_bracket2 = LNLinearAndLieBracket(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity, algebra_type='so3')
        
        self.fc_final = nn.Linear(feat_dim, 1, bias=False)

    def forward(self, x):
        '''
        x input of shape [B, F, 3, 1]
        '''

        x = self.ln_fc_bracket(x)  # [B, F, 3, 1]
        x = self.ln_fc(x)   # [B, F, 3, 1]
        x = self.ln_fc_bracket2(x)
        x = self.ln_fc2(x)

        x = torch.permute(x, (0, 3, 2, 1))  # [B, 1, 3, F]
        x_out = rearrange(self.fc_final(x), 'b 1 k 1 -> b k')   # [B, 3]
        # x_out = rearrange(self.ln_pooling(x), 'b 1 k 1 -> b k')   # [B, F, 1, 1]
        return x_out


class MLP(nn.Module):
    def __init__(self, in_channels):
        super(MLP, self).__init__()
        feat_dim = 1024
        self.fc = nn.Linear(in_channels, feat_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(feat_dim, feat_dim)
        self.fc3 = nn.Linear(feat_dim, feat_dim)
        
        self.fc4 = nn.Linear(feat_dim, feat_dim)
        self.fc_final = nn.Linear(feat_dim, 3)

    def forward(self, x):
        '''
        x input of shape [B, F, 3, 1]
        '''
        B, F, _, _ = x.shape
        x = torch.reshape(x, (B, -1))
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x_out = torch.reshape(self.fc_final(x), (B, 3))     # [B, 3]

        return x_out
