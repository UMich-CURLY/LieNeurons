import sys  # nopep8
sys.path.append('.')  # nopep8

import numpy as np
import torch

from core.lie_group_util import *
from core.lie_neurons_layers import *

if __name__ == "__main__":
    print("testing equivariant linear layer")

    # test equivariant linear layer
    x = torch.Tensor(np.random.rand(8))

    X = R8_to_sl3(x)

    x_new = sl3_to_R8(X)

    print("input: ", x)
    print("output: ", x_new)

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

    print(sl3_to_R8(E1))
    print(sl3_to_R8(E2))
    print(sl3_to_R8(E3))
    print(sl3_to_R8(E4))
    print(sl3_to_R8(E5))
    print(sl3_to_R8(E6))
    print(sl3_to_R8(E7))
    print(sl3_to_R8(E8))
