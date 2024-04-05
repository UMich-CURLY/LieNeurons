import sys  # nopep8
sys.path.append('.')  # nopep8

import numpy as np
import scipy.linalg
import torch

from core.lie_alg_util import *

if __name__ == "__main__":

    v = torch.Tensor([1,2,3,4,5,6,7,8,9,10])
    print("v: ", v)
    sp4_hatlayer = HatLayer(algebra_type='sp4')
    M = sp4_hatlayer(v)
    print("M: ", M)
    v2 = vee(M, algebra_type='sp4')
    print("v after vee: ", v2)

    rnd_scale=5
    v = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 1, 10)))
    M = sp4_hatlayer(v).squeeze(0).squeeze(0)
    
    zeros = torch.zeros(2, 2)
    I = torch.eye(2)
    # Construct the 4x4 matrix by combining the components
    omega = torch.cat((torch.cat((zeros, I), dim=1),
                    torch.cat((-I, zeros), dim=1)), dim=0)
    
    print("Checking if M satisfy the definition of sp(4): ")
    print(omega@M+M.T@omega)