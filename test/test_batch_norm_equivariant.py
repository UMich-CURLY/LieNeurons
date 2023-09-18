import sys
sys.path.append('.')

import numpy as np
import torch

from core.lie_group_util import *
from core.lie_neurons_layers import *


if __name__ == "__main__":
    print("testing equivariant linear layer")

    # test equivariant linear layer
    num_points = 100
    num_features = 10
    out_features = 3
    batch_size = 20

    x = torch.Tensor(np.random.rand(batch_size, num_features, 8, num_points)).reshape(
        batch_size, num_features, 8, num_points)
    y = torch.Tensor(np.random.rand(8))

    # SL(3) transformation
    Y = torch.linalg.matrix_exp(R8_to_sl3(y))

    model = LNLinearAndKillingNonLinearAndPooling(
        num_features, out_features, share_nonlinearity=True, use_batch_norm=True, dim=4)

    new_x = torch.zeros_like(x)
    for b in range(batch_size):
        for i in range(num_points):
            for j in range(num_features):
                X = R8_to_sl3(x[b, j, :, i])

                new_X = Y @ X @ Y.inverse()
                new_x[b, j, :, i] = sl3_to_R8(new_X)

    model.eval()
    with torch.no_grad():
        out_x = model(x)
        out_new_x = model(new_x)

    out_x_y_conjugate = torch.zeros_like(out_x)
    for b in range(batch_size):
        for i in range(out_features):
            out_X = R8_to_sl3(out_x[b, i, :])
            out_new_X = R8_to_sl3(out_new_x[b, i, :])

            out_X_Y_conjugate = Y @ out_X @ Y.inverse()
            out_x_y_conjugate[b, i, :] = sl3_to_R8(out_X_Y_conjugate)

    test_result = torch.allclose(
        out_new_x, out_x_y_conjugate, rtol=1e-4, atol=1e-4)

    print("out x[0,0,:]", out_x[0, 0, :])
    print("out x conjugate[0,0,:]: ", out_x_y_conjugate[0, 0, :])
    print("out new x[0,0,:]: ", out_new_x[0, 0, :])
    print("differences: ", out_x_y_conjugate[0, 0, :] - out_new_x[0, 0, :])

    print("The network is equivariant: ", test_result)
