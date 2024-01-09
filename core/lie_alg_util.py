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

    def forward(self, v):
        """
        v: a tensor of arbitrary shape with the last dimension of size k
        """
        return (v[..., None, None]*self.E_bases).sum(dim=-3)

def vee(M, algebra='sl3'):
    if algebra == 'so3':
        return vee_so3(M)
    elif algebra == 'sl3':
        return vee_sl3(M)

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

def killingform(x_hat, d_hat, algebra_type='sl3', feature_wise=False):
    if algebra_type == 'so3':
        return killingform_so3(x_hat, d_hat, feature_wise)
    elif algebra_type == 'sl3':
        return killingform_sl3(x_hat, d_hat, feature_wise)
    
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
    # return 6*torch.einsum('...ii', torch.matmul(x_hat,d_hat))[..., None]  # [B,F,N,1] equivalent to the above


class HatLayerSO3(torch.nn.Module):
    def __init__(self):
        super(HatLayerSO3, self).__init__()

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
            [Ex, Ey, Ez], dim=0)  # [8,3,3]
        self.register_buffer('E_bases', E_bases)

    def forward(self, v):
        """
        v: a tensor of arbitrary shape with the last dimension of size 3
        """

        # print("v",v.shape)
        # print("b",self.E_bases.shape)
        return (v[..., None, None]*self.E_bases).sum(dim=-3)
    
class HatLayerSl3(torch.nn.Module):
    def __init__(self):
        super(HatLayerSl3, self).__init__()

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

    def forward(self, v):
        """
        v: a tensor of arbitrary shape with the last dimension of size 8
        """

        return (v[..., None, None]*self.E_bases).sum(dim=-3)
