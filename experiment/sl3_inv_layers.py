
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


from core.lie_group_util import *
from core.lie_alg_util import *
from core.lie_neurons_layers import *


class SL3InvariantLayers(nn.Module):
    def __init__(self, in_channels):
        super(SL3InvariantLayers, self).__init__()
        self.ln_fc = LNLinearAndKillingRelu(in_channels, 10)
        self.ln_inv = LNInvariantPooling(method='killing')
        self.fc = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        '''
        x input of shape [B, F, 8, 1]
        '''
        x = self.ln_fc(x)   # [B, F, 8, 1]
        x_inv = self.ln_inv(x)  # [B, F, 1, 1]
        x_inv = self.fc(torch.permute(x_inv, (0, 3, 2, 1)))  # [B, 1, 1, 10]
        x_inv = self.relu(x_inv)    # [B, 1, 1, 10]
        x_out = torch.reshape(self.fc2(x_inv), (-1, 1))     # [B, 1]

        return x_out
