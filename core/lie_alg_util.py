import sys
import numpy as np
import torch
sys.path.append('.')


from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class HatLayer(torch.nn.Module):
    def __init__(self, algebra_type='sl3'):
        super(HatLayer, self).__init__()
        if algebra_type == 'so3':
            Ex = torch.Tensor([[0, 0, 0],
                            [0, 0, -1],
                            [0, 1, 0]])
            Ey = torch.Tensor([[0, 0, 1],
                            [0, 0, 0],
                            [-1, 0, 0]])
            Ez = torch.Tensor([[0, -1, 0],
                            [1, 0, 0],
                            [0, 0, 0]])

            E_bases = torch.stack(
                [Ex, Ey, Ez], dim=0)  # [3,3,3]
            self.register_buffer('E_bases', E_bases)

        elif algebra_type == 'sl3':
            E1 = torch.Tensor([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, 0]])
            E2 = torch.Tensor([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, 0]])
            E3 = torch.Tensor([[0, -1, 0],
                            [1, 0, 0],
                            [0, 0, 0]])
            E4 = torch.Tensor([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, -2]])
            E5 = torch.Tensor([[0, 0, 1],
                            [0, 0, 0],
                            [0, 0, 0]])
            E6 = torch.Tensor([[0, 0, 0],
                            [0, 0, 1],
                            [0, 0, 0]])
            E7 = torch.Tensor([[0, 0, 0],
                            [0, 0, 0],
                            [1, 0, 0]])
            E8 = torch.Tensor([[0, 0, 0],
                            [0, 0, 0],
                            [0, 1, 0]])
            E_bases = torch.stack(
                [E1, E2, E3, E4, E5, E6, E7, E8], dim=0)  # [8,3,3]
            
            self.register_buffer('E_bases', E_bases)

        elif algebra_type == 'se3':
            # we use the order of v = [t, \omega]^T

            E1 = torch.Tensor([[0, 0, 0, 1],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])
            E2 = torch.Tensor([[0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])
            E3 = torch.Tensor([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0]])
            E4 = torch.Tensor([[0, 0, 0, 0],
                            [0, 0, -1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 0]])
            E5 = torch.Tensor([[0, 0, 1, 0],
                            [0, 0, 0, 0],
                            [-1, 0, 0, 0],
                            [0, 0, 0, 0]])
            E6 = torch.Tensor([[0, -1, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])

            E_bases = torch.stack(
                [E1,E2,E3,E4,E5,E6], dim=0)  # [6,3,3]
            self.register_buffer('E_bases', E_bases)

        elif algebra_type == 'sp4':
            E1 = torch.tensor([[1, 0, 0, 0], 
                   [0, 0, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 0]])

            E2 = torch.tensor([[0, 1, 0, 0],
                            [0, 0, 0, 0], 
                            [0, 0, 0, 0],
                            [0, 0, -1, 0]])

            E3 = torch.tensor([[0, 0, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, -1],
                            [0, 0, 0, 0]])

            E4 = torch.tensor([[0, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, -1]])

            E5 = torch.tensor([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 0]])

            E6 = torch.tensor([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0]])

            E7 = torch.tensor([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 1, 0, 0]])

            E8 = torch.tensor([[0, 0, 1, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])

            E9 = torch.tensor([[0, 0, 0, 1],
                            [0, 0, 1, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])

            E10 = torch.tensor([[0, 0, 0, 0],
                                [0, 0, 0, 1],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0]])
            E_bases = torch.stack(
                [E1,E2,E3,E4,E5,E6,E7,E8,E9,E10], dim=0)  # [10,3,3]
            self.register_buffer('E_bases', E_bases)
        else:
            raise ValueError('Invalid algebra type for the hat operation')

    def forward(self, v):
        """
        v: a tensor of arbitrary shape with the last dimension of size k
        """
        return (v[..., None, None]*self.E_bases).sum(dim=-3)

def vee(M, algebra_type='sl3'):
    if algebra_type == 'so3':
        return vee_so3(M)
    elif algebra_type == 'sl3':
        return vee_sl3(M)
    elif algebra_type == 'se3':
        return vee_se3(M)
    elif algebra_type == 'sp4':
        return vee_sp4(M)
    else:
        raise ValueError('Invalid algebra type for the vee operation')

def vee_so3(M):
    # [0 , -z, y ]
    # [z ,  0, -x]
    # [-y,  x, 0 ]
    v = torch.zeros(M.shape[:-2]+(3,)).to(M.device)
    v[..., 0] = M[..., 2, 1]
    v[..., 1] = M[..., 0, 2]
    v[..., 2] = M[..., 1, 0]
    return v

def vee_sl3(M):
    # [a1 + a4, a2 - a3,    a5]
    # [a2 + a3, a4 - a1,    a6]
    # [     a7,      a8, -2*a4]
    v = torch.zeros(M.shape[:-2]+(8,)).to(M.device)
    v[..., 3] = -0.5*M[..., 2, 2]
    v[..., 4] = M[..., 0, 2]
    v[..., 5] = M[..., 1, 2]
    v[..., 6] = M[..., 2, 0]
    v[..., 7] = M[..., 2, 1]
    v[..., 0] = (M[..., 0, 0] - v[..., 3])

    v[..., 1] = 0.5*(M[..., 0, 1] + M[..., 1, 0])
    v[..., 2] = 0.5*(M[..., 1, 0] - M[..., 0, 1])
    return v

def vee_se3(M):
    # [0 ,  -wz,  wy,  tx]
    # [wz ,   0, -wx,  ty]
    # [-wy,  wx,   0,  tz]
    # [  0,   0,   0,   0]
    v = torch.zeros(M.shape[:-2]+(6,)).to(M.device)

    v[..., 0] = M[..., 0, 3]
    v[..., 1] = M[..., 1, 3]
    v[..., 2] = M[..., 2, 3]
    v[..., 3] = M[..., 2, 1]
    v[..., 4] = M[..., 0, 2]
    v[..., 5] = M[..., 1, 0]
    return v

def vee_sp4(M):
    v = torch.zeros(M.shape[:-2]+(10,)).to(M.device)

    v[..., 0] = M[..., 0, 0]
    v[..., 1] = M[..., 0, 1]
    v[..., 2] = M[..., 1, 0]
    v[..., 3] = M[..., 1, 1]
    v[..., 4] = M[..., 2, 0]
    v[..., 5] = M[..., 2, 1]
    v[..., 6] = M[..., 3, 1]
    v[..., 7] = M[..., 0, 2]
    v[..., 8] = M[..., 1, 2]
    v[..., 9] = M[..., 1, 3]
    return v

def killingform(x_hat, d_hat, algebra_type='sl3', feature_wise=False):
    if algebra_type == 'so3':
        return killingform_so3(x_hat, d_hat, feature_wise)
    elif algebra_type == 'sl3':
        return killingform_sl3(x_hat, d_hat, feature_wise)
    elif algebra_type == 'sp4':
        return killingform_sp4(x_hat, d_hat, feature_wise)
    else:
        raise ValueError('Invalid algebra type for the Killing form')
    
def killingform_so3(x_hat, d_hat, feature_wise=False):
    """
    x: a tensor of arbitrary shape with the last two dimension of size 3*3
    d: a tensor of arbitrary shape with the last two dimension of size 3*3
    killing form for so3 is tr(x_hat@d_hat)
    """

    if not feature_wise:
        return (x_hat.transpose(-1, -2)*d_hat).sum(dim=(-1, -2))[..., None]   # [B,F,N,1]
    else:
        return torch.einsum('...ii', torch.matmul(x_hat,d_hat))[..., None]
    
def killingform_sl3(x_hat, d_hat, feature_wise=False):
    """
    x: a tensor of arbitrary shape with the last two dimension of size 3*3
    d: a tensor of arbitrary shape with the last two dimension of size 3*3
    killing form for sl3 is 6tr(x_hat@d_hat)
    """
    if not feature_wise:
        return 6*(x_hat.transpose(-1, -2)*d_hat).sum(dim=(-1, -2))[..., None]   # [B,F,N,1]
    else:
        x_hat = rearrange(x_hat, 'b f n m1 m2 -> b f 1 n m1 m2')
        d_hat = rearrange(d_hat, 'b d n m1 m2 -> b 1 d n m1 m2')
        kf = 6*(x_hat.transpose(-1, -2)*d_hat).sum(dim=(-1, -2))
        kf = rearrange(kf, 'b f d n -> b (f d) n 1')
        
        return kf
def killingform_sp4(x_hat, d_hat, feature_wise=False):
    """
    x: a tensor of arbitrary shape with the last two dimension of size 4*4
    d: a tensor of arbitrary shape with the last two dimension of size 4*4
    killing form for sp4 is 6tr(x_hat@d_hat)
    """
    if not feature_wise:
        return 6*(x_hat.transpose(-1, -2)*d_hat).sum(dim=(-1, -2))[..., None]   # [B,F,N,1]
    else:
        return 6*(x_hat.transpose(-1, -2)*d_hat).sum(dim=(-1, -2))[..., None]   # [B,F,N,1]

def lie_bracket(x, y):
    return x@y - y@x