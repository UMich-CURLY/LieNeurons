import sys
import numpy as np
import torch
sys.path.append('.')

from core.lie_neurons_layers import *
from core.lie_group_util import *


class LNLinearAndKillingNonLinear(nn.Module):
    def __init__(self, in_channels, out_channels, share_nonlinearity=False):
        super(LNLinearAndKillingNonLinear, self).__init__()
        self.share_nonlinearity = share_nonlinearity

        self.linear = LNLinear(in_channels, out_channels)
        self.leaky_relu = LNKillingRelu(out_channels, share_nonlinearity=share_nonlinearity)
    
    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Conv
        x = self.linear(x)
        # LeakyReLU
        x_out = self.leaky_relu(x)
        return x_out


if __name__ == "__main__":
  print("testing equivariant linear layer")

  # test equivariant linear layer
  x = torch.Tensor(np.random.rand(8)).reshape(1,1,8,1)
  y = torch.Tensor(np.random.rand(8))
  
  model = LNLinearAndKillingNonLinear(1,1)

  X = R8_to_sl3(x[0,0,:,0])
  Y = torch.linalg.matrix_exp(R8_to_sl3(y))

  new_X = Y @ X @Y.inverse()
  new_x = sl3_to_R8(new_X)

  model.eval()
  with torch.no_grad():
    out_x = model(x)
    out_new_x = model(new_x.reshape(1,1,8,1))

  out_X = R8_to_sl3(out_x[0,0,:,0])
  out_new_X = R8_to_sl3(out_new_x[0,0,:,0])

  out_X_Y_congugate = Y @ out_X @ Y.inverse()
  out_x_y_conjugate = sl3_to_R8(out_X_Y_congugate)

  print("out X", out_X)
  print("out X congugate: ", out_X_Y_congugate)
  print("out new X: ", out_new_X)

  print("out x", out_x[0,0,:,0])
  print("out x congugate: ", out_x_y_conjugate)
  print("out new x: ", out_new_x[0,0,:,0])