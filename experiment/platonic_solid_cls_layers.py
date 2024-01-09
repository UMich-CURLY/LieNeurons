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


class LNPlatonicSolidClassifier(nn.Module):
    def __init__(self, in_channels):
        super(LNPlatonicSolidClassifier, self).__init__()
        feat_dim = 256
        share_nonlinearity = False
        self.ln_fc = LNLinearAndKillingRelu(
            in_channels, feat_dim, share_nonlinearity=share_nonlinearity)
        self.ln_fc2 = LNLinearAndKillingRelu(
            feat_dim, feat_dim, share_nonlinearity=share_nonlinearity)
        self.ln_fc3 = LNLinearAndKillingRelu(
            feat_dim, feat_dim, share_nonlinearity=share_nonlinearity)
        self.ln_pooling = LNMaxPool(
            feat_dim, abs_killing_form=False)  # [B, F, 8, 1]
        self.ln_inv = LNInvariant(feat_dim, method='self_killing')
        self.fc_final = nn.Linear(feat_dim, 3, bias=False)

    def forward(self, x):
        '''
        x input of shape [B, F, 8, 1]
        '''
        x = self.ln_fc(x)   # [B, F, 8, N]
        # x = self.ln_fc2(x)  # [B, F, 8, N]
        # x = self.ln_fc3(x)  # [B, F, 8, N]
        x = self.ln_pooling(x)  # [B, F, 8, 1]
        x_inv = self.ln_inv(x).unsqueeze(-1)  # [B, F, 1, 1]
        x_inv = torch.permute(x_inv, (0, 3, 2, 1))  # [B, 1, 1, F]
        x_out = rearrange(self.fc_final(x_inv),
                          'b 1 1 cls -> b cls')   # [B, cls]

        return x_out


class LNReluPlatonicSolidClassifier(nn.Module):
    def __init__(self, in_channels):
        super(LNReluPlatonicSolidClassifier, self).__init__()
        feat_dim = 256
        share_nonlinearity = False
        self.ln_fc = LNLinearAndKillingRelu(
            in_channels, feat_dim, share_nonlinearity=share_nonlinearity)
        self.ln_pooling = LNMaxPool(
            feat_dim, abs_killing_form=False)  # [B, F, 8, 1]
        self.ln_inv = LNInvariant(feat_dim, method='self_killing')
        self.fc_final = nn.Linear(feat_dim, 3, bias=False)

    def forward(self, x):
        '''
        x input of shape [B, F, 8, 1]
        '''
        x = self.ln_fc(x)   # [B, F, 8, N]
        x = self.ln_pooling(x)  # [B, F, 8, 1]
        x_inv = self.ln_inv(x).unsqueeze(-1)  # [B, F, 1, 1]
        x_inv = torch.permute(x_inv, (0, 3, 2, 1))  # [B, 1, 1, F]
        x_out = rearrange(self.fc_final(x_inv),
                          'b 1 1 cls -> b cls')   # [B, cls]

        return x_out


class LNBracketPlatonicSolidClassifier(nn.Module):
    def __init__(self, in_channels):
        super(LNBracketPlatonicSolidClassifier, self).__init__()
        feat_dim = 256
        share_nonlinearity = False
        self.ln_fc = LNLinearAndLieBracket(
            in_channels, feat_dim, share_nonlinearity=share_nonlinearity)
        # self.ln_fc2 = LNLinearAndLieBracket(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity)
        self.ln_pooling = LNMaxPool(
            feat_dim, abs_killing_form=False)  # [B, F, 8, 1]
        self.ln_inv = LNInvariant(feat_dim, method='self_killing')
        self.fc_final = nn.Linear(feat_dim, 3, bias=False)

    def forward(self, x):
        '''
        x input of shape [B, F, 8, 1]
        '''
        x = self.ln_fc(x)   # [B, F, 8, N]
        x = self.ln_pooling(x)  # [B, F, 8, 1]
        x_inv = self.ln_inv(x).unsqueeze(-1)  # [B, F, 1, 1]
        x_inv = torch.permute(x_inv, (0, 3, 2, 1))  # [B, 1, 1, F]
        x_out = rearrange(self.fc_final(x_inv),
                          'b 1 1 cls -> b cls')   # [B, cls]

        return x_out


class LNReluBracketPlatonicSolidClassifier(nn.Module):
    def __init__(self, in_channels):
        super(LNReluBracketPlatonicSolidClassifier, self).__init__()
        feat_dim = 256
        share_nonlinearity = False
        self.ln_fc = LNLinearAndKillingRelu(
            in_channels, feat_dim, share_nonlinearity=share_nonlinearity)
        self.ln_fc2 = LNLinearAndLieBracket(
            feat_dim, feat_dim, share_nonlinearity=share_nonlinearity)
        self.ln_pooling = LNMaxPool(
            feat_dim, abs_killing_form=False)  # [B, F, 8, 1]
        self.ln_inv = LNInvariant(feat_dim, method='self_killing')
        self.fc_final = nn.Linear(feat_dim, 3, bias=False)

    def forward(self, x):
        '''
        x input of shape [B, F, 8, 1]
        '''
        x = self.ln_fc(x)   # [B, F, 8, N]
        x = self.ln_fc2(x)  # [B, F, 8, N]
        x = self.ln_pooling(x)  # [B, F, 8, 1]
        x_inv = self.ln_inv(x).unsqueeze(-1)  # [B, F, 1, 1]
        x_inv = torch.permute(x_inv, (0, 3, 2, 1))  # [B, 1, 1, F]
        x_out = rearrange(self.fc_final(x_inv),
                          'b 1 1 cls -> b cls')   # [B, cls]

        return x_out


class LNBracketNoResidualConnectPlatonicSolidClassifier(nn.Module):
    def __init__(self, in_channels):
        super(LNBracketNoResidualConnectPlatonicSolidClassifier, self).__init__()
        feat_dim = 256
        share_nonlinearity = False
        self.ln_fc = LNLinearAndLieBracketNoResidualConnect(
            in_channels, feat_dim, share_nonlinearity=share_nonlinearity)
        # self.ln_fc2 = LNLinearAndLieBracket(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity)
        self.ln_pooling = LNMaxPool(
            feat_dim, abs_killing_form=False)  # [B, F, 8, 1]
        self.ln_inv = LNInvariant(feat_dim, method='self_killing')
        self.fc_final = nn.Linear(feat_dim, 3, bias=False)

    def forward(self, x):
        '''
        x input of shape [B, F, 8, 1]
        '''
        x = self.ln_fc(x)   # [B, F, 8, N]
        x = self.ln_pooling(x)  # [B, F, 8, 1]
        x_inv = self.ln_inv(x).unsqueeze(-1)  # [B, F, 1, 1]
        x_inv = torch.permute(x_inv, (0, 3, 2, 1))  # [B, 1, 1, F]
        x_out = rearrange(self.fc_final(x_inv),
                          'b 1 1 cls -> b cls')   # [B, cls]

        return x_out


class MLP(nn.Module):
    def __init__(self, in_channels):
        super(MLP, self).__init__()
        feat_dim = 256
        self.fc = nn.Linear(in_channels, feat_dim)
        self.fc2 = nn.Linear(feat_dim, feat_dim)
        self.fc3 = nn.Linear(feat_dim, feat_dim)
        self.relu = nn.ReLU()
        self.fc_final = nn.Linear(feat_dim, 3)

    def forward(self, x):
        '''
        x input of shape [B, F, 8, N]
        '''
        B, F, _, _ = x.shape
        x = torch.reshape(x, (B, -1))
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x_out = torch.reshape(self.fc_final(x), (B, 3))     # [B, cls]

        return x_out
