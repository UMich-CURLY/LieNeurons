import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('.')

from core.lie_group_util import *

EPS = 1e-6

class LNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LNLinear, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x):
        '''
        x: input of shape(B,F,8,N)
        '''
        x_out = self.fc(x.transpose(1,-1)).transpose(1,-1)
        return x_out

class LNKillingRelu(nn.Module):
    def __init__(self, in_channels, share_nonlinearity=False):
        super(LNKillingRelu,self).__init__()
        if share_nonlinearity == True:
            self.learned_dir = nn.Linear(in_channels, 1, bias=False)
        else:
            self.learned_dir = nn.Linear(in_channels, in_channels, bias=False)
    def forward(self, x):
        '''
        x: point features of shape [B, F, 8, N]
        '''
        B,F,_,N = x.shape
        x_out = torch.zeros_like(x)
        
        d = self.learned_dir(x.transpose(1,-1)).transpose(1,-1)
        for b in range(B):
            for f in range(F):
                for n in range(N):
                    # generate elements in sl3
                    X = R8_to_sl3(x[b,f,:,n])
                    D = R8_to_sl3(d[b,f,:,n])

                    killing_form = torch.trace(X@D)
                    print(killing_form)
                    if killing_form < 0:
                        x_out[b,f,:,n] = x[b,f,:,n]
                    else:
                        x_out[b,f,:,n] = x[b,f,:,n] - (-killing_form) * d[b,f,:,n]

        return x_out