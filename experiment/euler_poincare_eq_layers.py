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
        
    def forward(self, t, x):
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
        # print(y.shape)
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
        # print(y.shape)
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
        # print(y.shape)
        return self.net(y)