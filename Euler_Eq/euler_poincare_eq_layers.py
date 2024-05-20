import sys  # nopep8
sys.path.append('.')  # nopep8

import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from core.lie_neurons_layers import *
from core.lie_alg_util import *

def interpolate_u(t, u, Dt):
    ii = int(t // Dt)
    if ii >= len(u) - 1:
        ii = len(u) - 1
        u_t_interpolated = u[ii]
    else:
        u_t_interpolated = u[ii] + (t - ii * Dt) * (u[ii + 1] - u[ii]) / Dt
    return u_t_interpolated

class ODEMyTest(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
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

    def forward(self, t, x):
        '''
        x input of shape [B, F, 3, 1]
        '''
        x = rearrange(x,'b f d -> b f d 1')

        x = self.ln_fc_bracket(x)  # [B, F, 3, 1]
        x = self.ln_fc(x)   # [B, F, 3, 1]
        x = self.ln_fc_bracket2(x)
        x = self.ln_fc2(x)

        x = torch.permute(x, (0, 3, 2, 1))  # [B, 1, 3, F]
        x_out = rearrange(self.fc_final(x), 'b 1 k 1 -> b 1 k')  
        return x_out
    
    #     feat_dim = 1024
    #     share_nonlinearity = False
    #     leaky_relu = True
    #     self.ln_fc = LNLinearAndKillingRelu(
    #         in_channels, feat_dim, share_nonlinearity=share_nonlinearity, leaky_relu=leaky_relu, algebra_type='so3')
    #     self.ln_fc2 = LNLinearAndKillingRelu(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity,leaky_relu=leaky_relu, algebra_type='so3')
    #     self.ln_fc3 = LNLinearAndKillingRelu(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity,leaky_relu=leaky_relu, algebra_type='so3')
    #     self.ln_fc4 = LNLinearAndKillingRelu(feat_dim, feat_dim,share_nonlinearity=share_nonlinearity,leaky_relu=leaky_relu, algebra_type='so3')
        
    #     self.fc_final = nn.Linear(feat_dim, 1, bias=False)

        
    #     # self.ln_fc.modules()
    #     # for n in self.net.modules():
    #     #     if isinstance(n, nn.Linear):
    #     #         nn.init.normal_(n.weight, mean=0, std=0.1)

    # def forward(self, t, x):
    #     '''
    #     x input of shape [B, F, 3, 1]
    #     '''
    #     x = rearrange(x,'b f d -> b f d 1')

    #     x = self.ln_fc(x)   # [B, F, 3, 1]
    #     x = self.ln_fc2(x)  # [B, F, 3, 1]
    #     x = self.ln_fc3(x)
    #     x = self.ln_fc4(x)
        
    #     x = torch.permute(x, (0, 3, 2, 1))  # [B, 1, 3, F]
    #     x_out = rearrange(self.fc_final(x), 'b 1 k 1 -> b 1 k')   # [B, 3]
    #     return x_out

class LNODEFunc(nn.Module):

    def __init__(self, R=torch.eye(3), device='cpu'):
        super(LNODEFunc, self).__init__()

        self.num_m_feature = 20
        self.R = R.to(device)
        self.m = nn.Parameter(torch.zeros((3,self.num_m_feature)))
        self.map_m_to_m1 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m2 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m3 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.ln_bracket = LNLinearAndLieBracketChannelMix(1,20,algebra_type='so3',residual_connect=False)
        # self.ln_bracket2 = LNLinearAndLieBracket(20,1,algebra_type='so3')
        self.linear = LNLinear(20,1)

        nn.init.normal_(self.m, mean=0, std=0.1)
        # for n in self.net.modules():
        #     if isinstance(n, nn.Linear):
        #         nn.init.normal_(n.weight, mean=0, std=0.1)
                # nn.init.constant_(m.bias, val=0)
    def set_R(self,R):
        self.R = R.to(self.R.device)

    def forward(self, t, x):
        # print(y.shape)
        x_reshape = rearrange(x,'b f d -> b f d 1')
        # print(self.R.device)
        m1 = self.map_m_to_m1(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m2 = self.map_m_to_m2(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m3 = self.map_m_to_m3(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m1 = rearrange(m1,'1 f k 1 -> k f')
        m2 = rearrange(m2,'1 f k 1 -> k f')
        m3 = rearrange(m3,'1 f k 1 -> k f')
        M1 = torch.matmul(m1,m1.transpose(0,1))
        M2 = torch.matmul(m2,m2.transpose(0,1))
        M3 = torch.matmul(m3,m3.transpose(0,1))

        out = self.ln_bracket(x_reshape,M1,M2)  # [b f d 1]
        out = torch.einsum('d k, b f k n -> b f d n',M3,out)
        out = self.linear(out)
        out = rearrange(out,'b f d 1 -> b f d')
        return out


class LNODEFunc2(nn.Module):

    def __init__(self, R=torch.eye(3), device='cpu'):
        super(LNODEFunc2, self).__init__()

        self.num_m_feature = 20
        self.R = R.to(device)
        self.m = nn.Parameter(torch.zeros((3,self.num_m_feature)))
        self.map_m_to_m1 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m2 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m3 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.ln_bracket = LNLinearAndLieBracketChannelMix(1,20,algebra_type='so3',residual_connect=False)
        # self.ln_bracket2 = LNLinearAndLieBracket(20,1,algebra_type='so3')
        # self.linear = LNLinear(20,1)
        self.ln_relu = LNLinearAndKillingRelu(20,1,algebra_type='so3')

        nn.init.normal_(self.m, mean=0, std=0.1)
        # for n in self.net.modules():
        #     if isinstance(n, nn.Linear):
        #         nn.init.normal_(n.weight, mean=0, std=0.1)
                # nn.init.constant_(m.bias, val=0)
    def set_R(self,R):
        self.R = R.to(self.R.device)

    def forward(self, t, x):
        # print(y.shape)
        x_reshape = rearrange(x,'b f d -> b f d 1')
        # print(self.R.device)
        m1 = self.map_m_to_m1(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m2 = self.map_m_to_m2(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m3 = self.map_m_to_m3(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m1 = rearrange(m1,'1 f k 1 -> k f')
        m2 = rearrange(m2,'1 f k 1 -> k f')
        m3 = rearrange(m3,'1 f k 1 -> k f')
        M1 = torch.matmul(m1,m1.transpose(0,1))
        M2 = torch.matmul(m2,m2.transpose(0,1))
        M3 = torch.matmul(m3,m3.transpose(0,1))

        out = self.ln_bracket(x_reshape,M1,M2)  # [b f d 1]
        out = torch.einsum('d k, b f k n -> b f d n',M3,out)
        out = self.ln_relu(out)
        out = rearrange(out,'b f d 1 -> b f d')
        return out


class LNODEFunc3(nn.Module):

    def __init__(self, R=torch.eye(3), device='cpu'):
        super(LNODEFunc3, self).__init__()

        self.num_m_feature = 20
        self.R = R.to(device)
        self.m = nn.Parameter(torch.zeros((3,self.num_m_feature)))
        self.map_m_to_m1 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m2 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        # self.map_m_to_m3 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.ln_bracket = LNLinearAndLieBracketChannelMix(1,20,algebra_type='so3',residual_connect=False)
        # self.ln_bracket2 = LNLinearAndLieBracket(20,1,algebra_type='so3')
        # self.linear = LNLinear(20,1)
        self.ln_relu = LNLinearAndKillingRelu(20,1,algebra_type='so3')

        nn.init.normal_(self.m, mean=0, std=0.1)
        # for n in self.net.modules():
        #     if isinstance(n, nn.Linear):
        #         nn.init.normal_(n.weight, mean=0, std=0.1)
                # nn.init.constant_(m.bias, val=0)
    def set_R(self,R):
        self.R = R.to(self.R.device)

    def forward(self, t, x):
        # print(y.shape)
        x_reshape = rearrange(x,'b f d -> b f d 1')
        # print(self.R.device)
        m1 = self.map_m_to_m1(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m2 = self.map_m_to_m2(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        # m3 = self.map_m_to_m3(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m1 = rearrange(m1,'1 f k 1 -> k f')
        m2 = rearrange(m2,'1 f k 1 -> k f')
        # m3 = rearrange(m3,'1 f k 1 -> k f')
        M1 = torch.matmul(m1,m1.transpose(0,1))
        M2 = torch.matmul(m2,m2.transpose(0,1))
        # M3 = torch.matmul(m3,m3.transpose(0,1))

        out = self.ln_bracket(x_reshape,M1,M2)  # [b f d 1]
        # out = torch.einsum('d k, b f k n -> b f d n',M3,out)
        out = self.ln_relu(out)
        out = rearrange(out,'b f d 1 -> b f d')
        return out
    

class LNODEFunc4(nn.Module):

    def __init__(self, R=torch.eye(3), device='cpu'):
        super(LNODEFunc4, self).__init__()

        self.num_m_feature = 20
        self.R = R.to(device)
        self.m = nn.Parameter(torch.zeros((3,self.num_m_feature)))
        self.map_m_to_m1 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m2 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        # self.map_m_to_m3 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.ln_bracket = LNLinearAndLieBracketChannelMix(1,20,algebra_type='so3',residual_connect=False)
        # self.ln_bracket2 = LNLinearAndLieBracket(20,1,algebra_type='so3')
        self.linear = LNLinear(20,1)
        # self.ln_relu = LNLinearAndKillingRelu(20,1,algebra_type='so3')

        nn.init.normal_(self.m, mean=0, std=0.1)
        # for n in self.net.modules():
        #     if isinstance(n, nn.Linear):
        #         nn.init.normal_(n.weight, mean=0, std=0.1)
                # nn.init.constant_(m.bias, val=0)
    def set_R(self,R):
        self.R = R.to(self.R.device)

    def forward(self, t, x):
        # print(x.shape)
        x_reshape = rearrange(x,'b f d -> b f d 1')
        # print(self.R.device)
        m1 = self.map_m_to_m1(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m2 = self.map_m_to_m2(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        # m3 = self.map_m_to_m3(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m1 = rearrange(m1,'1 f k 1 -> k f')
        m2 = rearrange(m2,'1 f k 1 -> k f')
        # m3 = rearrange(m3,'1 f k 1 -> k f')
        M1 = torch.matmul(m1,m1.transpose(0,1))
        M2 = torch.matmul(m2,m2.transpose(0,1))
        # M3 = torch.matmul(m3,m3.transpose(0,1))
    
        out = self.ln_bracket(x_reshape,M1,M2)  # [b f d 1]
        # out = torch.einsum('d k, b f k n -> b f d n',M3,out)
        out = self.linear(out)
        out = rearrange(out,'b f d 1 -> b f d')
        return out

class LNODEFunc5(nn.Module):

    def __init__(self, R=torch.eye(3), device='cpu'):
        super(LNODEFunc5, self).__init__()

        self.num_m_feature = 30
        self.R = R.to(device)
        self.m = nn.Parameter(torch.zeros((3,self.num_m_feature)))
        self.map_m_to_m1 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m2 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        # self.map_m_to_m3 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.ln_relu = LNLinearAndKillingRelu(1,30,algebra_type='so3')
        self.ln_bracket = LNLinearAndLieBracketChannelMix(30,30,algebra_type='so3',residual_connect=False)
        # self.ln_bracket2 = LNLinearAndLieBracket(20,1,algebra_type='so3')
        # self.linear = LNLinear(20,1)
        self.ln_relu2 = LNLinearAndKillingRelu(30,1,algebra_type='so3')

        nn.init.normal_(self.m, mean=0, std=0.1)
        # for n in self.net.modules():
        #     if isinstance(n, nn.Linear):
        #         nn.init.normal_(n.weight, mean=0, std=0.1)
                # nn.init.constant_(m.bias, val=0)
    def set_R(self,R):
        self.R = R.to(self.R.device)

    def forward(self, t, x):
        # print(y.shape)
        x_reshape = rearrange(x,'b f d -> b f d 1')
        # print(self.R.device)
        m1 = self.map_m_to_m1(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m2 = self.map_m_to_m2(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        # m3 = self.map_m_to_m3(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m1 = rearrange(m1,'1 f k 1 -> k f')
        m2 = rearrange(m2,'1 f k 1 -> k f')
        # m3 = rearrange(m3,'1 f k 1 -> k f')
        M1 = torch.matmul(m1,m1.transpose(0,1))
        M2 = torch.matmul(m2,m2.transpose(0,1))
        # M3 = torch.matmul(m3,m3.transpose(0,1))
        out = self.ln_relu(x_reshape)
        out = self.ln_bracket(out,M1,M2)  # [b f d 1]
        # out = torch.einsum('d k, b f k n -> b f d n',M3,out)
        out = self.ln_relu2(out)
        out = rearrange(out,'b f d 1 -> b f d')
        return out
    
class LNODEFunc6(nn.Module):

    def __init__(self, R=torch.eye(3), device='cpu'):
        super(LNODEFunc6, self).__init__()

        self.num_m_feature = 20
        self.R = R.to(device)
        self.m = nn.Parameter(torch.zeros((3,self.num_m_feature)))
        self.map_m_to_m1 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m2 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        # self.map_m_to_m3 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.ln_bracket = LNLinearAndLieBracketChannelMix(1,20,algebra_type='so3',residual_connect=True)
        # self.ln_bracket2 = LNLinearAndLieBracket(20,1,algebra_type='so3')
        # self.linear = LNLinear(20,1)
        self.ln_relu = LNLinearAndKillingRelu(20,1,algebra_type='so3')

        nn.init.normal_(self.m, mean=0, std=0.1)
        # for n in self.net.modules():
        #     if isinstance(n, nn.Linear):
        #         nn.init.normal_(n.weight, mean=0, std=0.1)
                # nn.init.constant_(m.bias, val=0)
    def set_R(self,R):
        self.R = R.to(self.R.device)

    def forward(self, t, x):
        # print(y.shape)
        x_reshape = rearrange(x,'b f d -> b f d 1')
        # print(self.R.device)
        m1 = self.map_m_to_m1(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m2 = self.map_m_to_m2(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        # m3 = self.map_m_to_m3(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m1 = rearrange(m1,'1 f k 1 -> k f')
        m2 = rearrange(m2,'1 f k 1 -> k f')
        # m3 = rearrange(m3,'1 f k 1 -> k f')
        M1 = torch.matmul(m1,m1.transpose(0,1))
        M2 = torch.matmul(m2,m2.transpose(0,1))
        # M3 = torch.matmul(m3,m3.transpose(0,1))

        out = self.ln_bracket(x_reshape,M1,M2)  # [b f d 1]
        # out = torch.einsum('d k, b f k n -> b f d n',M3,out)
        out = self.ln_relu(out)
        out = rearrange(out,'b f d 1 -> b f d')
        return out

class LNODEFunc7(nn.Module):

    def __init__(self, R=torch.eye(3), device='cpu'):
        super(LNODEFunc7, self).__init__()

        self.num_m_feature = 20
        self.R = R.to(device)
        self.m = nn.Parameter(torch.zeros((3,self.num_m_feature)))
        self.map_m_to_m1 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m2 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        # self.map_m_to_m3 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.ln_bracket = LNLinearAndLieBracketChannelMix(1,20,algebra_type='so3',residual_connect=True)
        # self.ln_bracket2 = LNLinearAndLieBracket(20,1,algebra_type='so3')
        self.linear = LNLinear(20,1)
        # self.ln_relu = LNLinearAndKillingRelu(20,1,algebra_type='so3')

        nn.init.normal_(self.m, mean=0, std=0.1)
        # for n in self.net.modules():
        #     if isinstance(n, nn.Linear):
        #         nn.init.normal_(n.weight, mean=0, std=0.1)
                # nn.init.constant_(m.bias, val=0)
    def set_R(self,R):
        self.R = R.to(self.R.device)
        
    def forward(self, t, x):
        # print(x.shape)
        x_reshape = rearrange(x,'b f d -> b f d 1')
        # print(self.R.device)
        m1 = self.map_m_to_m1(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m2 = self.map_m_to_m2(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        # m3 = self.map_m_to_m3(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m1 = rearrange(m1,'1 f k 1 -> k f')
        m2 = rearrange(m2,'1 f k 1 -> k f')
        # m3 = rearrange(m3,'1 f k 1 -> k f')
        M1 = torch.matmul(m1,m1.transpose(0,1))
        M2 = torch.matmul(m2,m2.transpose(0,1))
        # M3 = torch.matmul(m3,m3.transpose(0,1))

        out = self.ln_bracket(x_reshape,M1,M2)  # [b f d 1]
        # out = torch.einsum('d k, b f k n -> b f d n',M3,out)
        out = self.linear(out)
        out = rearrange(out,'b f d 1 -> b f d')
        return out
    

    

class LNODEFunc8(nn.Module):
    def __init__(self, R=torch.eye(3), device='cpu'):
        super(LNODEFunc8, self).__init__()

        self.R = R.to(device)
        self.ln_bracket = LNLinearAndLieBracket(1,20,algebra_type='so3')
        self.linear = LNLinear(20,1)

    def set_R(self,R):
        self.R = R.to(self.R.device)
        
    def forward(self, t, x):
        # print(x.shape)
        x_reshape = rearrange(x,'b f d -> b f d 1')

        out = self.ln_bracket(x_reshape)  # [b f d 1]
        # out = torch.einsum('d k, b f k n -> b f d n',M3,out)
        out = self.linear(out)
        out = rearrange(out,'b f d 1 -> b f d')
        return out


class LNODEFunc9(nn.Module):

    def __init__(self, R=torch.eye(3), device='cpu'):
        super(LNODEFunc9, self).__init__()

        self.num_m_feature = 20
        self.R = R.to(device)
        self.m = nn.Parameter(torch.zeros((3,self.num_m_feature)))
        self.map_m_to_m1 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m2 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        # self.map_m_to_m3 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.ln_bracket = LNLinearAndLieBracketChannelMix(2,20,algebra_type='so3',residual_connect=True)
        # self.ln_bracket2 = LNLinearAndLieBracket(20,1,algebra_type='so3')
        self.linear = LNLinear(20,1)
        # self.ln_relu = LNLinearAndKillingRelu(20,1,algebra_type='so3')
        self.ufunc = lambda t: 10*torch.tensor([torch.sin(t),torch.cos(t),torch.sin(t)]).to(device)
        nn.init.normal_(self.m, mean=0, std=0.1)
        # for n in self.net.modules():
        #     if isinstance(n, nn.Linear):
        #         nn.init.normal_(n.weight, mean=0, std=0.1)
                # nn.init.constant_(m.bias, val=0)
    def set_R(self,R):
        self.R = R.to(self.R.device)
        
    def forward(self, t, x):
        # print(x.shape)
        x_reshape = rearrange(x,'b f d -> b f d 1')
        # print(type(t))

        u = torch.matmul(self.R,self.ufunc(t)).repeat(x_reshape.shape[0],1,1).unsqueeze(-1)
        x_reshape = torch.cat([x_reshape,u],dim=1)
        # print("--------")
        # print(x_reshape.shape)
        # print(u.shape)
        # print(self.R.device)
        m1 = self.map_m_to_m1(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m2 = self.map_m_to_m2(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        # m3 = self.map_m_to_m3(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m1 = rearrange(m1,'1 f k 1 -> k f')
        m2 = rearrange(m2,'1 f k 1 -> k f')
        # m3 = rearrange(m3,'1 f k 1 -> k f')
        M1 = torch.matmul(m1,m1.transpose(0,1))
        M2 = torch.matmul(m2,m2.transpose(0,1))
        # M3 = torch.matmul(m3,m3.transpose(0,1))

        out = self.ln_bracket(x_reshape,M1,M2)  # [b f d 1]
        # print("after bracket ",out.shape)
        # out = torch.einsum('d k, b f k n -> b f d n',M3,out)
        out = self.linear(out)
        out = rearrange(out,'b f d 1 -> b f d')
        return out

class LNODEFunc10(nn.Module):

    def __init__(self, R=torch.eye(3), device='cpu'):
        super(LNODEFunc10, self).__init__()

        self.num_m_feature = 20
        self.R = R.to(device)
        self.m = nn.Parameter(torch.zeros((3,self.num_m_feature)))
        self.map_m_to_m1 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m2 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m3 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m4 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        # self.map_m_to_m3 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.ln_bracket = LNLinearAndLieBracketChannelMix(1,20,algebra_type='so3',residual_connect=True)
        self.ln_bracket2 = LNLinearAndLieBracketChannelMix(1,20,algebra_type='so3',residual_connect=True)

        # self.ln_bracket2 = LNLinearAndLieBracket(20,1,algebra_type='so3')
        self.linear = LNLinear(20,1)
        # self.ln_relu = LNLinearAndKillingRelu(20,1,algebra_type='so3')
        self.ufunc = lambda t: 10*torch.tensor([torch.sin(t),torch.cos(t),torch.sin(t)]).to(device)
        nn.init.normal_(self.m, mean=0, std=0.1)
        # for n in self.net.modules():
        #     if isinstance(n, nn.Linear):
        #         nn.init.normal_(n.weight, mean=0, std=0.1)
                # nn.init.constant_(m.bias, val=0)
    def set_R(self,R):
        self.R = R.to(self.R.device)
        
    def forward(self, t, x, u , Dt):
        # print(x.shape)
        x_reshape = rearrange(x,'b f d -> b f d 1')
        # print(type(t))

        u = torch.matmul(self.R,self.ufunc(t)).repeat(x_reshape.shape[0],1,1).unsqueeze(-1)
        # x_reshape = torch.cat([x_reshape,u],dim=1)
        # print("--------")
        # print(x_reshape.shape)
        # print(u.shape)
        # print(self.R.device)
        m1 = self.map_m_to_m1(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m2 = self.map_m_to_m2(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        # m3 = self.map_m_to_m3(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m1 = rearrange(m1,'1 f k 1 -> k f')
        m2 = rearrange(m2,'1 f k 1 -> k f')
        # m3 = rearrange(m3,'1 f k 1 -> k f')
        M1 = torch.matmul(m1,m1.transpose(0,1))
        M2 = torch.matmul(m2,m2.transpose(0,1))
        # M3 = torch.matmul(m3,m3.transpose(0,1))

        x_out = self.ln_bracket(x_reshape,M1,M2)  # [b f d 1]

        m3 = self.map_m_to_m3(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m4 = self.map_m_to_m4(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m3 = rearrange(m3,'1 f k 1 -> k f')
        m4 = rearrange(m4,'1 f k 1 -> k f')
        M3 = torch.matmul(m3,m3.transpose(0,1))
        M4 = torch.matmul(m4,m4.transpose(0,1))
        u_out = self.ln_bracket2(u,M3,M4)
        out = x_out + u_out
        # print("after bracket ",out.shape)
        # out = torch.einsum('d k, b f k n -> b f d n',M3,out)
        out = self.linear(out)
        out = rearrange(out,'b f d 1 -> b f d')
        return out
    
class LNODEFunc11(nn.Module):

    def __init__(self, R=torch.eye(3), device='cpu'):
        super(LNODEFunc11, self).__init__()

        self.num_m_feature = 20
        self.R = R.to(device)
        self.m = nn.Parameter(torch.zeros((3,self.num_m_feature)))
        self.map_m_to_m1 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m2 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m3 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m4 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        # self.map_m_to_m3 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.ln_bracket = LNLinearAndLieBracketChannelMix(1,20,algebra_type='so3',residual_connect=True)
        self.ln_bracket2 = LNLinearAndLieBracketChannelMix(1,20,algebra_type='so3',residual_connect=True)

        # self.ln_bracket2 = LNLinearAndLieBracket(20,1,algebra_type='so3')
        self.linear = LNLinear(20,1)
        # self.ln_relu = LNLinearAndKillingRelu(20,1,algebra_type='so3')
        self.ufunc = lambda t: 10*torch.tensor([torch.sin(t),torch.cos(t),torch.sin(t)]).to(device)
        self.I = torch.Tensor([[12, -5., 7.],[-5., 20., -2.],[7., -2., 5.]]).unsqueeze(0).to(device)
        self.I_inv = torch.inverse(self.I)
        nn.init.normal_(self.m, mean=0, std=0.1)
        # for n in self.net.modules():
        #     if isinstance(n, nn.Linear):
        #         nn.init.normal_(n.weight, mean=0, std=0.1)
                # nn.init.constant_(m.bias, val=0)
    def set_R(self,R):
        self.R = R.to(self.R.device)
        
    def forward(self, t, x):
        # print(x.shape)
        x_reshape = rearrange(x,'b f d -> b f d 1')
        # print(type(t))

        u = torch.matmul(self.I_inv,torch.matmul(self.R,self.ufunc(t))).repeat(x_reshape.shape[0],1,1).unsqueeze(-1)
        # x_reshape = torch.cat([x_reshape,u],dim=1)
        # print("--------")
        # print(x_reshape.shape)
        # print(u.shape)
        # print(self.R.device)
        m1 = self.map_m_to_m1(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m2 = self.map_m_to_m2(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        # m3 = self.map_m_to_m3(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m1 = rearrange(m1,'1 f k 1 -> k f')
        m2 = rearrange(m2,'1 f k 1 -> k f')
        # m3 = rearrange(m3,'1 f k 1 -> k f')
        M1 = torch.matmul(m1,m1.transpose(0,1))
        M2 = torch.matmul(m2,m2.transpose(0,1))
        # M3 = torch.matmul(m3,m3.transpose(0,1))

        x_out = self.ln_bracket(x_reshape,M1,M2)  # [b f d 1]

        # m3 = self.map_m_to_m3(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        # m4 = self.map_m_to_m4(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        # m3 = rearrange(m3,'1 f k 1 -> k f')
        # m4 = rearrange(m4,'1 f k 1 -> k f')
        # M3 = torch.matmul(m3,m3.transpose(0,1))
        # M4 = torch.matmul(m4,m4.transpose(0,1))
        # u_out = self.ln_bracket2(u,M3,M4)
        out = x_out
        # print("after bracket ",out.shape)
        # out = torch.einsum('d k, b f k n -> b f d n',M3,out)
        out = self.linear(out) + u
        out = rearrange(out,'b f d 1 -> b f d')
        return out

class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )
        
        self.R = torch.eye(3)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def set_R(self,R):
        self.R = R.to(self.R.device)

    def forward(self, t, y):
        return self.net(y)
    
class ODEFunc2(nn.Module):

    def __init__(self):
        super(ODEFunc2, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )
        
        self.R = torch.eye(3)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def set_R(self,R):
        self.R = R.to(self.R.device)

    def forward(self, t, y):
        return self.net(y)
    
class ODEFunc3(nn.Module):

    def __init__(self):
        super(ODEFunc3, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )
        
        self.R = torch.eye(3)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def set_R(self,R):
        self.R = R.to(self.R.device)

    def forward(self, t, y):
        return self.net(y)
    

class ODEFunc_WithInput(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(6, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )
        
        self.R = torch.eye(3)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def set_R(self,R):
        self.R = R.to(self.R.device)

    def forward(self, t, y, u ,Dt):
        ut = interpolate_u(t, u, Dt)
        return self.net(torch.cat((y, ut), dim=y.dim() - 1))
    
class ODEFunc1_WithInput(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(6, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),
        )
        
        self.R = torch.eye(3)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def set_R(self,R):
        self.R = R.to(self.R.device)

    def forward(self, t, y, u ,Dt):
        ut = interpolate_u(t, u, Dt)
        return self.net(torch.cat((y, ut), dim=y.dim() - 1))
    
class LNODEFunc_WithInput(nn.Module):
    def __init__(self, R=torch.eye(3), device='cpu'):
        super().__init__()

        self.num_m_feature = 20
        self.R = R.to(device)
        self.m = nn.Parameter(torch.zeros((3,self.num_m_feature)))
        self.map_m_to_m1 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m2 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        # self.map_m_to_m3 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.ln_bracket = LNLinearAndLieBracketChannelMix(2,20,algebra_type='so3',residual_connect=False)
        # self.ln_bracket2 = LNLinearAndLieBracket(20,1,algebra_type='so3')
        self.linear = LNLinear(20,1)

        nn.init.normal_(self.m, mean=0, std=0.1)
        # for n in self.net.modules():
        #     if isinstance(n, nn.Linear):
        #         nn.init.normal_(n.weight, mean=0, std=0.1)
                # nn.init.constant_(m.bias, val=0)
    def set_R(self,R):
        self.R = R.to(self.R.device)

    def forward(self, t, x, u ,Dt):
        ut = interpolate_u(t, u, Dt)
        x_and_u = torch.cat([x, ut], dim=1)
        x_reshape = rearrange(x_and_u,'b f d -> b f d 1')
       
        m1 = self.map_m_to_m1(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m2 = self.map_m_to_m2(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))

        m1 = rearrange(m1,'1 f k 1 -> k f')
        m2 = rearrange(m2,'1 f k 1 -> k f')
        
        M1 = torch.matmul(m1,m1.transpose(0,1))
        M2 = torch.matmul(m2,m2.transpose(0,1))

        out = self.ln_bracket(x_reshape,M1,M2)  # [b f d 1]
        out = self.linear(out)
        out = rearrange(out,'b f d 1 -> b f d')
        return out
    

class LNODEFunc1_WithInput(nn.Module):
    def __init__(self, R=torch.eye(3), device='cpu'):
        super().__init__()

        self.num_m_feature = 20
        self.R = R.to(device)
        self.m = nn.Parameter(torch.zeros((3,self.num_m_feature)))
        self.map_m_to_m1 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m2 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        
        self.ln_bracket = LNLinearAndLieBracketChannelMix(1,20,algebra_type='so3',residual_connect=False)
        self.ln_bracket2 = LNLinearAndLieBracketChannelMix(1,20,algebra_type='so3',residual_connect=False)
        self.linear = LNLinear(20,1)

        nn.init.normal_(self.m, mean=0, std=0.1)
        # for n in self.net.modules():
        #     if isinstance(n, nn.Linear):
        #         nn.init.normal_(n.weight, mean=0, std=0.1)
                # nn.init.constant_(m.bias, val=0)
    def set_R(self,R):
        self.R = R.to(self.R.device)

    def forward(self, t, x, u ,Dt):
        ut = interpolate_u(t, u, Dt)
        ut_reshape = rearrange(ut,'b f d -> b f d 1')

        x_reshape = rearrange(x,'b f d -> b f d 1')
       
        m1 = self.map_m_to_m1(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m2 = self.map_m_to_m2(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))

        m1 = rearrange(m1,'1 f k 1 -> k f')
        m2 = rearrange(m2,'1 f k 1 -> k f')
        
        M1 = torch.matmul(m1,m1.transpose(0,1))
        M2 = torch.matmul(m2,m2.transpose(0,1))

        out = self.ln_bracket(x_reshape,M1,M2)  # [b f d 1]
        out1 = self.ln_bracket2(ut_reshape,M1,M2)
        out = out + out1
        out = self.linear(out)
        out = rearrange(out,'b f d 1 -> b f d')
        return out

class LNODEFunc2_WithInput(nn.Module):
    def __init__(self, R=torch.eye(3), device='cpu'):
        super().__init__()

        self.num_m_feature = 20
        self.R = R.to(device)
        self.m = nn.Parameter(torch.zeros((3,self.num_m_feature)))
        self.map_m_to_m1 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m2 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        
        self.ln_bracket = LNLinearAndLieBracketChannelMix(1,20,algebra_type='so3',residual_connect=True)
        self.ln_bracket2 = LNLinearAndLieBracketChannelMix(1,20,algebra_type='so3',residual_connect=True)
        self.linear = LNLinear(20,1)

        nn.init.normal_(self.m, mean=0, std=0.1)
        # for n in self.net.modules():
        #     if isinstance(n, nn.Linear):
        #         nn.init.normal_(n.weight, mean=0, std=0.1)
                # nn.init.constant_(m.bias, val=0)
    def set_R(self,R):
        self.R = R.to(self.R.device)

    def forward(self, t, x, u ,Dt):
        ut = interpolate_u(t, u, Dt)
        ut_reshape = rearrange(ut,'b f d -> b f d 1')

        x_reshape = rearrange(x,'b f d -> b f d 1')
       
        m1 = self.map_m_to_m1(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m2 = self.map_m_to_m2(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))

        m1 = rearrange(m1,'1 f k 1 -> k f')
        m2 = rearrange(m2,'1 f k 1 -> k f')
        
        M1 = torch.matmul(m1,m1.transpose(0,1))
        M2 = torch.matmul(m2,m2.transpose(0,1))

        out = self.ln_bracket(x_reshape,M1,M2)  # [b f d 1]
        out1 = self.ln_bracket2(ut_reshape,M1,M2)
        out = out + out1
        out = self.linear(out)
        out = rearrange(out,'b f d 1 -> b f d')
        return out
    
class LNODEfunc3_WithInput(nn.Module):
    def __init__(self, R=torch.eye(3), device='cpu'):
        super().__init__()

        self.num_m_feature = 20
        self.R = R.to(device)
        self.m = nn.Parameter(torch.zeros((3,self.num_m_feature)))
        self.map_m_to_m1 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m2 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m3 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m4 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        
        self.ln_bracket = LNLinearAndLieBracketChannelMix(1,20,algebra_type='so3',residual_connect=True)
        self.ln_bracket2 = LNLinearAndLieBracketChannelMix(1,20,algebra_type='so3',residual_connect=True)

        
        self.linear = LNLinear(20,1)
        
        nn.init.normal_(self.m, mean=0, std=0.1)

        # self.ufunc = lambda t: torch.tensor([[torch.sin(t),torch.cos(t),torch.sin(t)]]).to(device)
        
    def set_R(self,R):
        self.R = R.to(self.R.device)
        
    def forward(self, t, x, u , Dt):
        ut = interpolate_u(t, u, Dt)
        # # debug
        # print(ut.shape)
        # ut1 = self.ufunc(t)
        # print(ut1.shape)
        u_reshape = rearrange(ut,'b f d -> b f d 1')
        x_reshape = rearrange(x,'b f d -> b f d 1')


        m1 = self.map_m_to_m1(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m2 = self.map_m_to_m2(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        # m3 = self.map_m_to_m3(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m1 = rearrange(m1,'1 f k 1 -> k f')
        m2 = rearrange(m2,'1 f k 1 -> k f')
        
        M1 = torch.matmul(m1,m1.transpose(0,1))
        M2 = torch.matmul(m2,m2.transpose(0,1))
        

        x_out = self.ln_bracket(x_reshape,M1,M2)  # [b f d 1]

        m3 = self.map_m_to_m3(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m4 = self.map_m_to_m4(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m3 = rearrange(m3,'1 f k 1 -> k f')
        m4 = rearrange(m4,'1 f k 1 -> k f')
        M3 = torch.matmul(m3,m3.transpose(0,1))
        M4 = torch.matmul(m4,m4.transpose(0,1))

        u_out = self.ln_bracket2(u_reshape,M3,M4)
        out = x_out + u_out

        out = self.linear(out)
        out = rearrange(out,'b f d 1 -> b f d')
        return out
    
class LNODEfunc4_WithInput(nn.Module):
    def __init__(self, R=torch.eye(3), device='cpu'):
        super().__init__()

        self.num_m_feature = 20
        self.R = R.to(device)
        self.m = nn.Parameter(torch.zeros((3,self.num_m_feature)))
        self.map_m_to_m1 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m2 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m3 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m4 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        
        self.ln_bracket = LNLinearAndLieBracketChannelMix(1,20,algebra_type='so3',residual_connect=False)
        self.ln_bracket2 = LNLinearAndLieBracketChannelMix(1,20,algebra_type='so3',residual_connect=False)

        
        self.linear = LNLinear(20,1)
        
        nn.init.normal_(self.m, mean=0, std=0.1)
        
    def set_R(self,R):
        self.R = R.to(self.R.device)
        
    def forward(self, t, x, u , Dt):
        ut = interpolate_u(t, u, Dt)
        u_reshape = rearrange(ut,'b f d -> b f d 1')
        x_reshape = rearrange(x,'b f d -> b f d 1')


        m1 = self.map_m_to_m1(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m2 = self.map_m_to_m2(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        # m3 = self.map_m_to_m3(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m1 = rearrange(m1,'1 f k 1 -> k f')
        m2 = rearrange(m2,'1 f k 1 -> k f')
        
        M1 = torch.matmul(m1,m1.transpose(0,1))
        M2 = torch.matmul(m2,m2.transpose(0,1))
        

        x_out = self.ln_bracket(x_reshape,M1,M2)  # [b f d 1]

        m3 = self.map_m_to_m3(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m4 = self.map_m_to_m4(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m3 = rearrange(m3,'1 f k 1 -> k f')
        m4 = rearrange(m4,'1 f k 1 -> k f')
        M3 = torch.matmul(m3,m3.transpose(0,1))
        M4 = torch.matmul(m4,m4.transpose(0,1))

        u_out = self.ln_bracket2(u_reshape,M3,M4)
        out = x_out + u_out

        out = self.linear(out)
        out = rearrange(out,'b f d 1 -> b f d')
        return out
    
class LNODEfunc5_WithInput(nn.Module):
    def __init__(self, R=torch.eye(3), device='cpu'):
        super().__init__()

        self.num_m_feature = 20
        self.R = R.to(device)
        self.m = nn.Parameter(torch.zeros((3,self.num_m_feature)))
        self.map_m_to_m1 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m2 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m3 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m4 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        
        self.ln_bracket = LNLinearAndLieBracketChannelMix(1,20,algebra_type='so3',residual_connect=True)
        self.ln_bracket2 = LNLinearAndLieBracketChannelMix(1,20,algebra_type='so3',residual_connect=True)

        self.linear = LNLinear(40,1)
        
        nn.init.normal_(self.m, mean=0, std=0.1)
        
    def set_R(self,R):
        self.R = R.to(self.R.device)
        
    def forward(self, t, x, u , Dt):
        ut = interpolate_u(t, u, Dt)
        u_reshape = rearrange(ut,'b f d -> b f d 1')
        x_reshape = rearrange(x,'b f d -> b f d 1')


        m1 = self.map_m_to_m1(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m2 = self.map_m_to_m2(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        # m3 = self.map_m_to_m3(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m1 = rearrange(m1,'1 f k 1 -> k f')
        m2 = rearrange(m2,'1 f k 1 -> k f')
        
        M1 = torch.matmul(m1,m1.transpose(0,1))
        M2 = torch.matmul(m2,m2.transpose(0,1))
        

        x_out = self.ln_bracket(x_reshape,M1,M2)  # [b f d 1]

        m3 = self.map_m_to_m3(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m4 = self.map_m_to_m4(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m3 = rearrange(m3,'1 f k 1 -> k f')
        m4 = rearrange(m4,'1 f k 1 -> k f')
        M3 = torch.matmul(m3,m3.transpose(0,1))
        M4 = torch.matmul(m4,m4.transpose(0,1))

        u_out = self.ln_bracket2(u_reshape,M3,M4)
        out = torch.cat([x_out,u_out],dim=1)

        out = self.linear(out)
        out = rearrange(out,'b f d 1 -> b f d')
        return out
    
class LNODEfunc6_WithInput(nn.Module):
    def __init__(self, R=torch.eye(3), device='cpu'):
        super().__init__()

        self.num_m_feature = 20
        self.R = R.to(device)
        self.m = nn.Parameter(torch.zeros((3,self.num_m_feature)))
        self.m_u = nn.Parameter(torch.zeros((3,self.num_m_feature)))
        self.map_m_to_m1 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m2 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m3 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        self.map_m_to_m4 = LNLinearAndKillingRelu(self.num_m_feature,3,algebra_type='so3')
        
        self.ln_bracket = LNLinearAndLieBracketChannelMix(1,20,algebra_type='so3',residual_connect=True)
        self.ln_bracket2 = LNLinearAndLieBracketChannelMix(1,20,algebra_type='so3',residual_connect=True)

        
        self.linear = LNLinear(20,1)
        
        nn.init.normal_(self.m, mean=0, std=0.1)
        nn.init.normal_(self.m_u, mean=0, std=0.1)
        
    def set_R(self,R):
        self.R = R.to(self.R.device)
        
    def forward(self, t, x, u , Dt):
        ut = interpolate_u(t, u, Dt)
        # # debug
        # print(ut.shape)
        # ut1 = self.ufunc(t)
        # print(ut1.shape)
        u_reshape = rearrange(ut,'b f d -> b f d 1')
        x_reshape = rearrange(x,'b f d -> b f d 1')


        m1 = self.map_m_to_m1(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m2 = self.map_m_to_m2(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        # m3 = self.map_m_to_m3(rearrange(torch.matmul(self.R,self.m),'k f -> 1 f k 1'))
        m1 = rearrange(m1,'1 f k 1 -> k f')
        m2 = rearrange(m2,'1 f k 1 -> k f')
        
        M1 = torch.matmul(m1,m1.transpose(0,1))
        M2 = torch.matmul(m2,m2.transpose(0,1))
        

        x_out = self.ln_bracket(x_reshape,M1,M2)  # [b f d 1]

        m3 = self.map_m_to_m3(rearrange(torch.matmul(self.R,self.m_u),'k f -> 1 f k 1'))
        m4 = self.map_m_to_m4(rearrange(torch.matmul(self.R,self.m_u),'k f -> 1 f k 1'))
        m3 = rearrange(m3,'1 f k 1 -> k f')
        m4 = rearrange(m4,'1 f k 1 -> k f')
        M3 = torch.matmul(m3,m3.transpose(0,1))
        M4 = torch.matmul(m4,m4.transpose(0,1))

        u_out = self.ln_bracket2(u_reshape,M3,M4)
        out = x_out + u_out

        out = self.linear(out)
        out = rearrange(out,'b f d 1 -> b f d')
        return out
    
import warnings
def model_choose(model_type = 'neural_ode', device='cpu'):
    if device == 'cpu':
        warnings.warn("Using CPU!")

    match model_type:
        case 'LN_ode':
            func = LNODEFunc(device=device).to(device)
        case 'LN_ode2':
            func = LNODEFunc2(device=device).to(device)
        case 'LN_ode3':
            func = LNODEFunc3(device=device).to(device)
        case 'LN_ode4':
            func = LNODEFunc4(device=device).to(device)
        case 'LN_ode5':
            func = LNODEFunc5(device=device).to(device)
        case 'LN_ode6':
            func = LNODEFunc6(device=device).to(device)
        case 'LN_ode7':
            func = LNODEFunc7(device=device).to(device)
        case 'LN_ode8':
            func = LNODEFunc8(device=device).to(device)
        case 'LN_ode9':
            func = LNODEFunc9(device=device).to(device)
        case 'LN_ode10':
            func = LNODEFunc10(device=device).to(device) 
        case 'LN_ode11':
            func = LNODEFunc11(device=device).to(device)
        case 'neural_ode':
            func = ODEFunc().to(device)  
        case 'neural_ode2':
            func = ODEFunc2().to(device)
        case 'neural_ode3':
            func = ODEFunc3().to(device)
        # TEST
        case 'ODEMyTest':
            func = ODEMyTest(1).to(device)
        #With Input
        case 'LN_ode_WithInput':
            func = LNODEFunc_WithInput(device=device).to(device)
        case 'LN_ode_WithInput1':
            func = LNODEFunc1_WithInput(device=device).to(device)
        case 'LN_ode_WithInput2':
            func = LNODEFunc2_WithInput(device=device).to(device)
        case 'LN_ode_WithInput3':
            func = LNODEfunc3_WithInput(device=device).to(device)
        case 'LN_ode_WithInput4':
            func = LNODEfunc4_WithInput(device=device).to(device)
        case 'LN_ode_WithInput5':
            func = LNODEfunc5_WithInput(device=device).to(device)
        case 'LN_ode_WithInput6':
            func = LNODEfunc6_WithInput(device=device).to(device)
        case 'neural_ode_WithInput':
            func = ODEFunc_WithInput().to(device)
        case 'neural_ode_WithInput1':
            func = ODEFunc1_WithInput().to(device)
        case _  :
            raise KeyError("model_type not found")
    return func