import sys  # nopep8
sys.path.append('.')  # nopep8

import numpy as np
import scipy.linalg
import torch

from core.lie_alg_util import *

if __name__ == "__main__":

    v = torch.Tensor([1,2,3,4,5,6])
    print("v: ", v)
    se3_hatlayer = HatLayer(algebra_type='se3')
    M = se3_hatlayer(v)
    print("M: ", M)
    v2 = vee(M, algebra_type='se3')
    print("v after vee: ", v2)

    rnd_scale=5
    v = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 1, 6)))
    print("v: ", v)
    print(v.shape)
    M = se3_hatlayer(v)
    M_SE3 = scipy.linalg.expm(M[0,0,:,:].numpy())
    print("M_SE3: ", M_SE3)
    M2 = scipy.linalg.logm(M_SE3)

    print("M: ", M)
    print("M log: ", M2)

    v = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 1, 3)))
    print("v: ", v)
    print(v.shape)
    so3_hatlayer = HatLayer(algebra_type='so3')
    M = so3_hatlayer(v)
    print(M.shape)
    M_SE3 = scipy.linalg.expm(M[0,0,:,:].numpy())
    print("M_SE3: ", M_SE3)
    M2 = scipy.linalg.logm(M_SE3)

    print("M: ", M)
    print("M log: ", M2)