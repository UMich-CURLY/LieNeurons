import sys
import numpy as np
import torch
sys.path.append('.')

from core.lie_neurons_layers import *
from core.lie_group_util import *




if __name__ == "__main__":
  print("testing equivariant linear layer")

  # test equivariant linear layer
  x = torch.Tensor(np.random.rand(8))

  X = R8_to_sl3(x)

  x_new = sl3_to_R8(X)

  print("input: ", x)
  print("output: ", x_new)