import sys  # nopep8
sys.path.append('.')  # nopep8

import numpy as np
import scipy.linalg
import torch
import math

from core.lie_alg_util import *
from core.lie_group_util import *

if __name__ == "__main__":


    v1 = torch.rand(1,3)
    v1 = v1/torch.norm(v1)
    phi = math.pi
    v1 = phi*v1
    # print("v: ", v1)

    v2 = v1
    v2 = v2/torch.norm(v2)
    phi2 = math.pi
    v2 = phi2*v2

    v1 = torch.Tensor([[1.4911, 0.6458, 1.0547]])
    v2 = torch.Tensor([[0.2295, 2.0104, 0.0430]])

    so3_hatlayer = HatLayer(algebra_type='so3')
    K1 = so3_hatlayer(v1)
    K2 = so3_hatlayer(v2)

    R1 = exp_so3(K1[0,:,:])
    R2 = exp_so3(K2[0,:,:])

    print("v1: ", v1)
    print("v2: ", v2)
    print("R1: ", R1)
    print("R2: ", R2)

    R3 = R1@R2
    print("R3:", R3)
    K3 = log_SO3(R3)
    K3_BCH = BCH_approx(K1[0,:,:],K2[0,:,:])
    K3_BCH_SO3 = BCH_so3(K1[0,:,:],K2[0,:,:])
    print("K1: ", K1)
    print("K2: ", K2)
    print("----")
    print("K3: ", K3)
    print("K3 BCH: ", K3_BCH)
    print("K1+K2: ", K1+K2) 
    print("K3 BCH SO3: ", K3_BCH_SO3)
    print("norm K3: ", torch.norm(vee(K3, algebra_type='so3')))