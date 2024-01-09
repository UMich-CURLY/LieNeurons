import sys  # nopep8
sys.path.append('.')  # nopep8

import numpy as np
import torch

from core.lie_neurons_layers import *
from core.lie_alg_util import *
from experiment.sl3_inv_layers import *


if __name__ == "__main__":
    print("testing the invariant layer")

    # test equivariant linear layer
    num_points = 1
    num_features = 10
    out_features = 3
    batch_size = 20
    rnd_scale = 0.5

    hat_layer = HatLayerSl3()

    x = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (batch_size, num_features, 8, num_points))).reshape(
        batch_size, num_features, 8, num_points)
    y = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, 8))

    # SL(3) transformation
    Y = torch.linalg.matrix_exp(hat_layer(y))
    
    # model = LNInvariant(num_features,method='learned_killing')
    model = SL3InvariantLayers(num_features)

    x_hat = hat_layer(x.transpose(2, -1))
    new_x_hat = torch.matmul(Y, torch.matmul(x_hat, torch.inverse(Y)))
    new_x = vee_sl3(new_x_hat).transpose(2, -1)
    
    model.eval()
    with torch.no_grad():
        out_x = model(x)
        out_new_x = model(new_x)

    test_result = torch.allclose(
        out_new_x, out_x, rtol=1e-4, atol=1e-4)

    print("out x[0,0,:]", out_x[0, :])
    print("out new x[0,0,:]: ", out_new_x[0, :])
    print("differences: ", out_x[ 0, :] - out_new_x[ 0, :])

    print("The network is equivariant: ", test_result)
