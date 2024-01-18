import sys  # nopep8
sys.path.append('.')  # nopep8

import numpy as np
import scipy.linalg
import torch
import math

from core.lie_alg_util import *
from core.lie_group_util import *

if __name__ == "__main__":

    rnd_scale=1
    v = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 1, 3)))
    v = v/torch.norm(v)
    phi = math.pi*torch.rand(1)
    v = phi*v
    print("v: ", v)
    so3_hatlayer = HatLayer(algebra_type='so3')
    K = so3_hatlayer(v)
    K_SO3 = torch.asarray(scipy.linalg.expm(K[0,0,:,:].numpy()))
    K_SO3_2 = exp_hat_and_so3(v.T)
    K_SO3_3 = exp_so3(K[0,0,:,:])
    K_SO3_exp_gpu = exp_so3(K[0,0,:,:].to('cuda'))
    print("-------------------")
    print("SO3: ", K_SO3)
    print("SO3_2: ", K_SO3_2)
    print("SO3_3: ", K_SO3_3)
    print("-------------------")
    print("det SO3: ", np.linalg.det(K_SO3))
    print("det SO3_2: ", np.linalg.det(K_SO3_2))
    print("-------------------")
    K_after = log_SO3(K_SO3)
    K2_after = log_SO3(K_SO3_2)
    K_log_gpu = log_SO3(K_SO3_3.to('cuda'))
    K_exp_gpu_after = log_SO3(K_SO3_exp_gpu)

    print("K: ", K)
    print("M: ", K_after)
    print("M log: ", K2_after)
    print("M log gpu: ", K_exp_gpu_after)
    print("M log gpu: ", K_log_gpu)