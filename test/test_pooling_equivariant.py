import sys  # nopep8
sys.path.append('.')  # nopep8

import numpy as np
import torch

from core.lie_neurons_layers import *
from core.lie_alg_util import *

if __name__ == "__main__":
    print("testing equivariant linear layer")

    # test equivariant linear layer
    num_points = 100
    num_features = 10
    out_features = 3

    x = torch.Tensor(np.random.rand(num_features, 8, num_points)
                     ).reshape(1, num_features, 8, num_points)
    y = torch.Tensor(np.random.rand(8))

    hat_layer = HatLayer()

    # SL(3) transformation
    Y = torch.linalg.matrix_exp(hat_layer(y))

    model = LNLinearAndKillingReluAndPooling(
        num_features, out_features, share_nonlinearity=True, abs_killing_form=False)

    x_hat = hat_layer(x.transpose(2, -1))
    new_x_hat = torch.matmul(Y, torch.matmul(x_hat, torch.inverse(Y)))
    new_x = vee_sl3(new_x_hat).transpose(2, -1)

    model.eval()
    with torch.no_grad():
        out_x = model(x)
        out_new_x = model(new_x)

    out_x_hat = hat_layer(out_x.transpose(2, -1))
    out_x_hat_conj = torch.matmul(Y, torch.matmul(out_x_hat, torch.inverse(Y)))
    out_x_conj = vee_sl3(out_x_hat_conj).transpose(2, -1)

    test_result = torch.allclose(
        out_new_x, out_x_conj, rtol=1e-4, atol=1e-4)

    print("out x[0,0,:]", out_x[0, 0, :])
    print("out x conj[0,0,:]: ", out_x_conj[0, 0, :])
    print("out new x[0,0,:]: ", out_new_x[0, 0, :])
    print("differences: ", out_x_conj[0, 0, :] - out_new_x[0, 0, :])

    print("The network is equivariant: ", test_result)
