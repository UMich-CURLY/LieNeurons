import sys
import numpy as np
import torch
sys.path.append('.')

from core.lie_neurons_layers import *
from core.lie_group_util import *



if __name__ == "__main__":
  print("testing equivariant linear layer")

  # test equivariant linear layer
  num_points = 100
  num_features = 10
  out_features = 3

  x = torch.Tensor(np.random.rand(num_features,8,num_points)).reshape(1,num_features,8,num_points)
  y = torch.Tensor(np.random.rand(8))
  
  # SL(3) transformation
  Y = torch.linalg.matrix_exp(R8_to_sl3(y))
  
  model = LNLinearAndKillingNonLinearAndPooling(num_features,out_features)

  new_x = torch.zeros_like(x)
  for i in range(num_points):
    for j in range(num_features):
      X = R8_to_sl3(x[0,j,:,i])

      new_X = Y @ X @ Y.inverse()
      new_x[:,j,:,i] = sl3_to_R8(new_X)
  
  print("x_shape ",x.shape)
  print("new_x_shape ",new_x.shape)


  model.eval()
  with torch.no_grad():
    out_x = model(x)
    out_new_x = model(new_x)

  print(out_x.shape)
  out_x_y_conjugate = torch.zeros_like(out_x)
  for i in range(out_features):
    out_X = R8_to_sl3(out_x[0,i,:])
    out_new_X = R8_to_sl3(out_new_x[0,i,:])

    out_X_Y_conjugate = Y @ out_X @ Y.inverse()
    out_x_y_conjugate[0,i,:] = sl3_to_R8(out_X_Y_conjugate)

  print("out X", out_X)
  print("out X conjugate: ", out_X_Y_conjugate)
  print("out new X: ", out_new_X)

  print("out x", out_x)
  print("out x conjugate: ", out_x_y_conjugate)
  print("out new x: ", out_new_x)