import sys
import numpy as np
import torch
sys.path.append('.')

def R8_to_sl3(v):
    E1 = torch.Tensor([[1,0,0],
                   [0, -1, 0],
                   [0,0,0]])
    E2 = torch.Tensor([[0, 1,0],
            [1, 0 ,0],
            [0, 0 ,0]])
    E3 = torch.Tensor([[0, -1, 0],
            [1, 0, 0],
            [0, 0, 0]])
    E4 = torch.Tensor([[1, 0, 0],
            [0, 1, 0],
            [0, 0, -2]])
    E5 = torch.Tensor([[0,0,1],
            [0,0,0],
            [0,0,0]])
    E6 = torch.Tensor([[0,0,0],
            [0,0,1],
            [0,0,0]])
    E7 = torch.Tensor([[0,0,0],
            [0,0,0],
            [1,0,0]])
    E8 = torch.Tensor([[0,0,0],
            [0,0,0],
            [0,1,0]])
    return (v[0]*E1+v[1]*E2+v[2]*E3+v[3]*E4+v[4]*E5+v[5]*E6+v[6]*E7+v[7]*E8).to(v.device)

def sl3_to_R8(M):
    # [a1 + a4, a2 - a3,    a5]
    # [a2 + a3, a4 - a1,    a6]
    # [     a7,      a8, -2*a4]
    v = torch.zeros(8).to(M.device)
    v[3] = -0.5*M[2,2]
    v[4] = M[0,2]
    v[5] = M[1,2]
    v[6] = M[2,0]
    v[7] = M[2,1]
    v[0] = (M[0,0] - v[3])

    v[1] = 0.5*(M[0,1] + M[1,0])
    v[2] = 0.5*(M[1,0] - M[0,1])
    return v