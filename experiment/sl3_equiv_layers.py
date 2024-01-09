
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


class SL3EquivariantLayers(nn.Module):
    def __init__(self, in_channels):
        super(SL3EquivariantLayers, self).__init__()
        feat_dim = 256
        share_nonlinearity = False
        leaky_relu = True
        # self.ln_fc = LNLinearAndKillingRelu(
        #     in_channels, feat_dim, share_nonlinearity=share_nonlinearity, leaky_relu=leaky_relu)
        # # # self.ln_norm = LNBatchNorm(feat_dim, dim=4, affine=False, momentum=0.1)
        # self.ln_fc2 = LNLinearAndKillingRelu(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity,leaky_relu=leaky_relu)
        self.ln_fc3 = LNLinearAndKillingRelu(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity,leaky_relu=leaky_relu)
        # self.ln_fc4 = LNLinearAndKillingRelu(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity,leaky_relu=leaky_relu)
        # self.ln_fc5 = LNLinearAndKillingRelu(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity,leaky_relu=leaky_relu)
        # self.ln_fc6 = LNLinearAndKillingRelu(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity,leaky_relu=leaky_relu)
        self.ln_fc = LNLinearAndLieBracket(in_channels, feat_dim,share_nonlinearity=share_nonlinearity)
        self.ln_fc2 = LNLinearAndLieBracket(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity)
        # self.ln_fc4 = LNLinearAndLieBracket(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity)
        # self.fc = nn.Linear(feat_dim, feat_dim)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(feat_dim, feat_dim)
        # self.fc3 = nn.Linear(feat_dim, feat_dim)
        self.fc_final = nn.Linear(feat_dim, 1, bias=False)
        # self.ln_pooling = LNMaxPool(feat_dim, abs_killing_form = False) # [B, F, 8, 1]

    def forward(self, x):
        '''
        x input of shape [B, F, 8, 1]
        '''
        x = self.ln_fc(x)   # [B, F, 8, 1]
        # x = self.ln_norm(x)  # [B, F, 8, 1]
        x = self.ln_fc2(x)  # [B, F, 8, 1]
        # # x = self.ln_norm(x)  # [B, F, 8, 1]
        x = self.ln_fc3(x)  # [B, F, 8, 1]

        # x = self.ln_fc4(x)  # [B, F, 8, 1]

        # x = self.ln_fc5(x)  # [B, F, 8, 1]


        # x = self.ln_fc6(x)  # [B, F, 8, 1]
        x = torch.permute(x, (0, 3, 2, 1))  # [B, 1, 8, F]
        x_out = rearrange(self.fc_final(x), 'b 1 k 1 -> b k')   # [B, 8]
        # x_out = rearrange(self.ln_pooling(x), 'b 1 k 1 -> b k')   # [B, F, 1, 1]
        return x_out


class SL3EquivariantReluLayers(nn.Module):
    def __init__(self, in_channels):
        super(SL3EquivariantReluLayers, self).__init__()
        feat_dim = 256
        share_nonlinearity = False
        leaky_relu = True
        self.ln_fc = LNLinearAndKillingRelu(
            in_channels, feat_dim, share_nonlinearity=share_nonlinearity, leaky_relu=leaky_relu)
        self.ln_fc2 = LNLinearAndKillingRelu(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity,leaky_relu=leaky_relu)
        
        self.fc_final = nn.Linear(feat_dim, 1, bias=False)

    def forward(self, x):
        '''
        x input of shape [B, F, 8, 1]
        '''
        x = self.ln_fc(x)   # [B, F, 8, 1]
        x = self.ln_fc2(x)  # [B, F, 8, 1]
        
        x = torch.permute(x, (0, 3, 2, 1))  # [B, 1, 8, F]
        x_out = rearrange(self.fc_final(x), 'b 1 k 1 -> b k')   # [B, 8]
        return x_out


class SL3EquivariantBracketLayers(nn.Module):
    def __init__(self, in_channels):
        super(SL3EquivariantBracketLayers, self).__init__()
        feat_dim = 256
        share_nonlinearity = False
        leaky_relu = True
        self.ln_fc = LNLinearAndLieBracket(in_channels, feat_dim,share_nonlinearity=share_nonlinearity)
        self.ln_fc2 = LNLinearAndLieBracket(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity)
        
        self.fc_final = nn.Linear(feat_dim, 1, bias=False)

    def forward(self, x):
        '''
        x input of shape [B, F, 8, 1]
        '''
        x = self.ln_fc(x)   # [B, F, 8, 1]
        x = self.ln_fc2(x)  # [B, F, 8, 1]

        x = torch.permute(x, (0, 3, 2, 1))  # [B, 1, 8, F]
        x_out = rearrange(self.fc_final(x), 'b 1 k 1 -> b k')   # [B, 8]
        return x_out

class SL3EquivariantReluBracketLayers(nn.Module):
    def __init__(self, in_channels):
        super(SL3EquivariantReluBracketLayers, self).__init__()
        feat_dim = 256
        share_nonlinearity = False
        leaky_relu = True
        self.ln_fc = LNLinearAndKillingRelu(
            feat_dim, feat_dim, share_nonlinearity=share_nonlinearity, leaky_relu=leaky_relu)
        self.ln_fc2 = LNLinearAndKillingRelu(
            feat_dim, feat_dim, share_nonlinearity=share_nonlinearity, leaky_relu=leaky_relu)
        self.ln_fc_bracket = LNLinearAndLieBracket(in_channels, feat_dim,share_nonlinearity=share_nonlinearity)
        self.ln_fc_bracket2 = LNLinearAndLieBracket(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity)
        
        self.fc_final = nn.Linear(feat_dim, 1, bias=False)

    def forward(self, x):
        '''
        x input of shape [B, F, 8, 1]
        '''

        x = self.ln_fc_bracket(x)  # [B, F, 8, 1]
        x = self.ln_fc(x)   # [B, F, 8, 1]
        x = self.ln_fc_bracket2(x)
        x = self.ln_fc2(x)

        x = torch.permute(x, (0, 3, 2, 1))  # [B, 1, 8, F]
        x_out = rearrange(self.fc_final(x), 'b 1 k 1 -> b k')   # [B, 8]
        # x_out = rearrange(self.ln_pooling(x), 'b 1 k 1 -> b k')   # [B, F, 1, 1]
        return x_out

class SL3EquivariantBracketNoResidualConnectLayers(nn.Module):
    def __init__(self, in_channels):
        super(SL3EquivariantBracketNoResidualConnectLayers, self).__init__()
        feat_dim = 256
        share_nonlinearity = False
        leaky_relu = True
        self.ln_fc = LNLinearAndLieBracketNoResidualConnect(in_channels, feat_dim,share_nonlinearity=share_nonlinearity)
        self.ln_fc2 = LNLinearAndLieBracketNoResidualConnect(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity)
        
        self.fc_final = nn.Linear(feat_dim, 1, bias=False)

    def forward(self, x):
        '''
        x input of shape [B, F, 8, 1]
        '''
        x = self.ln_fc(x)   # [B, F, 8, 1]
        x = self.ln_fc2(x)  # [B, F, 8, 1]

        x = torch.permute(x, (0, 3, 2, 1))  # [B, 1, 8, F]
        x_out = rearrange(self.fc_final(x), 'b 1 k 1 -> b k')   # [B, 8]
        return x_out
    
class SL3InvariantLayersTest(nn.Module):
    def __init__(self, in_channels):
        super(SL3InvariantLayersTest, self).__init__()
        feat_dim = 256
        self.ln_fc = LNLinearAndKillingRelu(
            in_channels, feat_dim, share_nonlinearity=True)
        self.ln_norm = LNBatchNorm(feat_dim, dim=3)
        self.ln_fc2 = LNLinearAndKillingRelu(feat_dim, feat_dim)
        self.ln_fc3 = LNLinearAndKillingRelu(feat_dim, feat_dim)
        self.ln_inv = LNInvariant(method='killing')
        self.fc_in = nn.Linear(in_channels, feat_dim)
        self.fc = nn.Linear(feat_dim, feat_dim)
        self.fc_no_inv = nn.Linear(8*feat_dim, feat_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(feat_dim, feat_dim)
        self.fc3 = nn.Linear(feat_dim, feat_dim)
        self.fc_final_no_inv = nn.Linear(8*feat_dim, 1)
        self.fc_final = nn.Linear(feat_dim, 1)

    def forward(self, x):
        '''
        x input of shape [B, F, 8, 1]
        '''
        # test invariant layer + fc
        # x_inv = self.ln_inv(x)  # [B, F, 1, 1]
        # # [B, 1, 1, feat_dim]
        # x_inv = self.fc_in(torch.permute(x_inv, (0, 3, 2, 1)))
        # x_inv = self.relu(x_inv)    # [B, 1, 1, feat_dim]
        # x_inv = self.fc2(x_inv)
        # x_inv = self.relu(x_inv)
        # x_out = torch.reshape(self.fc_final(x_inv), (-1, 1))     # [B, 1]

        # test LN fc + fc
        # x = self.ln_fc(x)   # [B, F, 8, 1]
        # x = self.ln_fc2(x)   # [B, F, 8, 1]
        # # [B, 1, 1, feat_dim]
        # B, _, _, _ = x.shape
        # x_inv = torch.reshape(x, (B, -1))
        # x_inv = self.fc_no_inv(x_inv)
        # x_inv = self.relu(x_inv)    # [B, 1, 1, feat_dim]
        # x_inv = self.fc2(x_inv)
        # x_inv = self.relu(x_inv)
        # x_out = torch.reshape(self.fc_final(x_inv), (-1, 1))     # [B, 1]

        # test LN fc
        x = self.ln_fc(x)   # [B, F, 8, 1]
        x = self.ln_fc2(x)   # [B, F, 8, 1]
        x = self.ln_fc3(x)   # [B, F, 8, 1]

        # [B, 1, 1, feat_dim]
        B, _, _, _ = x.shape
        x_inv = torch.reshape(x, (B, -1))
        # x_inv = self.fc_no_inv(x_inv)
        # x_inv = self.relu(x_inv)    # [B, 1, 1, feat_dim]
        # x_inv = self.fc2(x_inv)
        # x_inv = self.relu(x_inv)
        x_out = torch.reshape(self.fc_final_no_inv(
            x_inv), (-1, 1))     # [B, 1]

        # test LN fc + fc
        # x = self.ln_fc(x)   # [B, F, 8, 1]
        # # x = self.ln_norm(x)  # [B, F, 8, 1]
        # # x = self.ln_fc2(x)  # [B, F, 8, 1]
        # x_inv = self.ln_inv(x)  # [B, F, 1, 1]
        # # [B, 1, 1, feat_dim]
        # # B, _, _, _ = x.shape
        # # x_inv = torch.reshape(x, (B, -1))
        # # x_inv = self.fc(x_inv)
        # # x_inv = self.fc_in(torch.permute(x_inv, (0, 3, 2, 1)))
        # x_inv = self.fc(torch.permute(x_inv, (0, 3, 2, 1)))
        # x_inv = self.relu(x_inv)    # [B, 1, 1, feat_dim]
        # x_inv = self.fc2(x_inv)
        # x_inv = self.relu(x_inv)
        # # x_inv = self.fc3(x_inv)
        # # x_inv = self.relu(x_inv)
        # x_out = torch.reshape(self.fc_final(x_inv), (-1, 1))     # [B, 1]

        return x_out


class MLP(nn.Module):
    def __init__(self, in_channels):
        super(MLP, self).__init__()
        feat_dim = 512
        self.fc = nn.Linear(in_channels, feat_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(feat_dim, feat_dim)
        self.fc3 = nn.Linear(feat_dim, feat_dim)
        self.fc_final = nn.Linear(feat_dim, 8)

    def forward(self, x):
        '''
        x input of shape [B, F, 8, 1]
        '''
        B, F, _, _ = x.shape
        x = torch.reshape(x, (B, -1))
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x_out = torch.reshape(self.fc_final(x), (B, 8))     # [B, 8]

        return x_out
