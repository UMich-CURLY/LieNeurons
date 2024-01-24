import sys  # nopep8
sys.path.append('.')  # nopep8

import numpy as np
import torch
from scipy.linalg import expm


from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from core.lie_neurons_layers import *


def invariant_function(x1, x2, x3, x4, x5):
    return torch.sin(torch.trace(x1@x2@x3))+torch.cos(torch.trace(x3@x4@x5))\
        -torch.pow(torch.trace(x5@x1), 6)/2.0+torch.det(x3@x2)+torch.exp(torch.trace(x4@x1))\
        +torch.trace(x1@x2@x3@x4@x5)


if __name__ == "__main__":
    data_saved_path = "data/sl3_inv_5_input_data/"
    data_name = "sl3_inv_1000_s_05_test"
    num_training = 1000
    num_testing = 1000
    num_conjugate = 500
    rnd_scale = 0.5

    train_data = {}
    test_data = {}

    hat_layer = HatLayer(algebra_type='sl3')

    x1 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 8, num_training)))\
        .reshape(1, 1, 8, num_training)
    x2 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 8, num_training)))\
        .reshape(1, 1, 8, num_training)
    x3 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 8, num_training)))\
        .reshape(1, 1, 8, num_training)
    x4 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 8, num_training)))\
        .reshape(1, 1, 8, num_training)
    x5 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 8, num_training)))\
        .reshape(1, 1, 8, num_training)

    # conjugate transformation
    h = torch.Tensor(np.random.uniform(-rnd_scale,
                     rnd_scale, (num_conjugate, num_training, 8)))
    H = torch.linalg.matrix_exp(hat_layer(h))
    # conjugate x1
    x1_hat = hat_layer(x1.transpose(2, -1))
    conj_x1_hat = torch.matmul(H, torch.matmul(x1_hat, torch.inverse(H)))
    conj_x1 = rearrange(vee_sl3(conj_x1_hat), 'b c t l -> b l t c')

    # conjugate x2
    x2_hat = hat_layer(x2.transpose(2, -1))
    conj_x2_hat = torch.matmul(H, torch.matmul(x2_hat, torch.inverse(H)))
    conj_x2 = rearrange(vee_sl3(conj_x2_hat), 'b c t l -> b l t c')

    # conjugate x3
    x3_hat = hat_layer(x3.transpose(2, -1))
    conj_x3_hat = torch.matmul(H, torch.matmul(x3_hat, torch.inverse(H)))
    conj_x3 = rearrange(vee_sl3(conj_x3_hat), 'b c t l -> b l t c')

    # conjugate x4
    x4_hat = hat_layer(x4.transpose(2, -1))
    conj_x4_hat = torch.matmul(H, torch.matmul(x4_hat, torch.inverse(H)))
    conj_x4 = rearrange(vee_sl3(conj_x4_hat), 'b c t l -> b l t c')

    # conjugate x5
    x5_hat = hat_layer(x5.transpose(2, -1))
    conj_x5_hat = torch.matmul(H, torch.matmul(x5_hat, torch.inverse(H)))
    conj_x5 = rearrange(vee_sl3(conj_x5_hat), 'b c t l -> b l t c')

 

    inv_output = torch.zeros((1, num_training, 1))
    # compute invariant function
    for n in range(num_training):
        inv_output[0, n, 0] = invariant_function(
            x1_hat[0, 0, n, :, :], x2_hat[0, 0, n, :, :],x3_hat[0, 0, n, :, :],\
                x4_hat[0, 0, n, :, :],x5_hat[0, 0, n, :, :])

        # print("-------------------------------")
        # print(inv_output[0,n,0])
        # for i in range(num_conjugate):
        #     print(invariant_function(conj_x1_hat[0, i, n, :, :],conj_x2_hat[0, i, n, :, :],\
        #                              conj_x3_hat[0, i, n, :, :],conj_x4_hat[0, i, n, :, :],\
        #                              conj_x5_hat[0, i, n, :, :]))

    train_data['x1'] = x1.numpy().reshape(8, num_training)
    train_data['x2'] = x2.numpy().reshape(8, num_training)
    train_data['x3'] = x3.numpy().reshape(8, num_training)
    train_data['x4'] = x4.numpy().reshape(8, num_training)
    train_data['x5'] = x5.numpy().reshape(8, num_training)
    train_data['x1_conjugate'] = conj_x1.numpy().reshape(8, num_training, num_conjugate)
    train_data['x2_conjugate'] = conj_x2.numpy().reshape(8, num_training, num_conjugate)
    train_data['x3_conjugate'] = conj_x3.numpy().reshape(8, num_training, num_conjugate)
    train_data['x4_conjugate'] = conj_x4.numpy().reshape(8, num_training, num_conjugate)
    train_data['x5_conjugate'] = conj_x5.numpy().reshape(8, num_training, num_conjugate)
    train_data['y'] = inv_output.numpy().reshape(1, num_training)

    np.savez(data_saved_path + data_name + "_train_data.npz", **train_data)


    '''
    Generate testing data
    '''
    x1 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 8, num_training)))\
        .reshape(1, 1, 8, num_testing)
    x2 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 8, num_testing)))\
        .reshape(1, 1, 8, num_testing)
    x3 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 8, num_testing)))\
        .reshape(1, 1, 8, num_testing)
    x4 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 8, num_testing)))\
        .reshape(1, 1, 8, num_testing)
    x5 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 8, num_testing)))\
        .reshape(1, 1, 8, num_testing)

    # conjugate transformation
    h = torch.Tensor(np.random.uniform(-rnd_scale,
                     rnd_scale, (num_conjugate, num_testing, 8)))
    H = torch.linalg.matrix_exp(hat_layer(h))
    # conjugate x1
    x1_hat = hat_layer(x1.transpose(2, -1))
    conj_x1_hat = torch.matmul(H, torch.matmul(x1_hat, torch.inverse(H)))
    conj_x1 = rearrange(vee_sl3(conj_x1_hat), 'b c t l -> b l t c')

    # conjugate x2
    x2_hat = hat_layer(x2.transpose(2, -1))
    conj_x2_hat = torch.matmul(H, torch.matmul(x2_hat, torch.inverse(H)))
    conj_x2 = rearrange(vee_sl3(conj_x2_hat), 'b c t l -> b l t c')

    # conjugate x3
    x3_hat = hat_layer(x3.transpose(2, -1))
    conj_x3_hat = torch.matmul(H, torch.matmul(x3_hat, torch.inverse(H)))
    conj_x3 = rearrange(vee_sl3(conj_x3_hat), 'b c t l -> b l t c')

    # conjugate x4
    x4_hat = hat_layer(x4.transpose(2, -1))
    conj_x4_hat = torch.matmul(H, torch.matmul(x4_hat, torch.inverse(H)))
    conj_x4 = rearrange(vee_sl3(conj_x4_hat), 'b c t l -> b l t c')

    # conjugate x5
    x5_hat = hat_layer(x5.transpose(2, -1))
    conj_x5_hat = torch.matmul(H, torch.matmul(x5_hat, torch.inverse(H)))
    conj_x5 = rearrange(vee_sl3(conj_x5_hat), 'b c t l -> b l t c')

    inv_output = torch.zeros((1, num_testing, 1))
    # compute invariant function
    for n in range(num_testing):
        inv_output[0, n, 0] = invariant_function(
            x1_hat[0, 0, n, :, :], x2_hat[0, 0, n, :, :],x3_hat[0, 0, n, :, :],\
                x4_hat[0, 0, n, :, :],x5_hat[0, 0, n, :, :])
        
        # print("-------------------------------")
        # print(inv_output[0,n,0])
        # for i in range(num_conjugate):
        #     print(invariant_function(conj_x1_hat[0, i, n, :, :],conj_x2_hat[0, i, n, :, :],\
        #                              conj_x3_hat[0, i, n, :, :],conj_x4_hat[0, i, n, :, :],\
        #                              conj_x5_hat[0, i, n, :, :]))

    test_data['x1'] = x1.numpy().reshape(8, num_testing)
    test_data['x2'] = x2.numpy().reshape(8, num_testing)
    test_data['x3'] = x3.numpy().reshape(8, num_testing)
    test_data['x4'] = x4.numpy().reshape(8, num_testing)
    test_data['x5'] = x5.numpy().reshape(8, num_testing)
    test_data['x1_conjugate'] = conj_x1.numpy().reshape(8, num_testing, num_conjugate)
    test_data['x2_conjugate'] = conj_x2.numpy().reshape(8, num_testing, num_conjugate)
    test_data['x3_conjugate'] = conj_x3.numpy().reshape(8, num_testing, num_conjugate)
    test_data['x4_conjugate'] = conj_x4.numpy().reshape(8, num_testing, num_conjugate)
    test_data['x5_conjugate'] = conj_x5.numpy().reshape(8, num_testing, num_conjugate)
    test_data['y'] = inv_output.numpy().reshape(1, num_testing)

    np.savez(data_saved_path + data_name + "_test_data.npz", **test_data)

    print("Done! Data saved to: \n", data_saved_path +
          data_name + "_train_data.npz\n", data_saved_path + data_name + "_test_data.npz")
