import sys  # nopep8
sys.path.append('.')  # nopep8

import numpy as np
import torch
from scipy.linalg import expm

from core.lie_neurons_layers import *


def invariant_function(x1, x2):
    return torch.sin(torch.trace(x1@x1))+torch.cos(torch.trace(x2@x2))\
        -torch.pow(torch.trace(x2@x2), 3)/2.0+torch.det(x1@x2)+torch.exp(torch.trace(x1@x1))


if __name__ == "__main__":
    data_saved_path = "data/sp4_inv_data/"
    data_name = "sp4_inv_10000_s_05_augmented"
    gen_augmented_training_data = True 
    num_training = 5000
    num_testing = 10000
    num_conjugate = 1
    rnd_scale = 0.5

    train_data = {}
    test_data = {}

    hat_layer = HatLayer(algebra_type='sp4')

    x1 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 10, num_training)))\
        .reshape(1, 1, 10, num_training)
    x2 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 10, num_training)))\
        .reshape(1, 1, 10, num_training)

    # conjugate transformation
    h = torch.Tensor(np.random.uniform(-rnd_scale,
                     rnd_scale, (num_conjugate, num_training, 10)))
    H = torch.linalg.matrix_exp(hat_layer(h))
    # conjugate x1
    x1_hat = hat_layer(x1.transpose(2, -1))
    conj_x1_hat = torch.matmul(H, torch.matmul(x1_hat, torch.inverse(H)))
    conj_x1 = rearrange(vee_sp4(conj_x1_hat), 'b c t l -> b l t c')

    # conjugate x2
    x2_hat = hat_layer(x2.transpose(2, -1))
    conj_x2_hat = torch.matmul(H, torch.matmul(x2_hat, torch.inverse(H)))
    conj_x2 = rearrange(vee_sp4(conj_x2_hat), 'b c t l -> b l t c')

    inv_output = torch.zeros((1, num_training, 1))
    # compute invariant function
    for n in range(num_training):
        inv_output[0, n, 0] = invariant_function(
            x1_hat[0, 0, n, :, :], x2_hat[0, 0, n, :, :])

        # print("--------------invariant function output: ------------------")
        # print(invariant_function(x1_hat[0, 0, n, :, :], x2_hat[0, 0, n, :, :]))
        # print(invariant_function(
        #     conj_x1_hat[0, 0, n, :, :], conj_x2_hat[0, 0, n, :, :]))
    
    # print(x1.shape)
    # print(conj_x1.shape)
    if(gen_augmented_training_data):
        # this is for training data only
        train_data['x1'] = torch.cat((x1.reshape(10, num_training),conj_x1.reshape(10, num_training*num_conjugate)),dim=1).numpy()
        train_data['x2'] = torch.cat((x2.reshape(10, num_training),conj_x2.reshape(10, num_training*num_conjugate)),dim=1).numpy()
        train_data['x1_conjugate'] = conj_x1.reshape(10, num_training, num_conjugate).repeat(1,num_conjugate+1,1).numpy()
        train_data['x2_conjugate'] = conj_x2.reshape(10, num_training, num_conjugate).repeat(1,num_conjugate+1,1).numpy()
        train_data['y'] = inv_output.reshape(1, num_training).repeat(1,num_conjugate+1).numpy()
    else:
        train_data['x1'] = x1.numpy().reshape(10, num_training)
        train_data['x2'] = x2.numpy().reshape(10, num_training)
        train_data['x1_conjugate'] = conj_x1.numpy().reshape(10, num_training, num_conjugate)
        train_data['x2_conjugate'] = conj_x2.numpy().reshape(10, num_training, num_conjugate)
        train_data['y'] = inv_output.numpy().reshape(1, num_training)

    np.savez(data_saved_path + data_name + "_train_data.npz", **train_data)


    '''
    Generate testing data
    '''
    x1 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 10, num_testing)))\
        .reshape(1, 1, 10, num_testing)
    x2 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 10, num_testing)))\
        .reshape(1, 1, 10, num_testing)

    # conjugate transformation
    h = torch.Tensor(np.random.uniform(-rnd_scale,
                     rnd_scale, (num_conjugate, num_testing, 10)))
    H = torch.linalg.matrix_exp(hat_layer(h))
    # conjugate x1
    x1_hat = hat_layer(x1.transpose(2, -1))
    conj_x1_hat = torch.matmul(H, torch.matmul(x1_hat, torch.inverse(H)))
    conj_x1 = rearrange(vee_sp4(conj_x1_hat), 'b c t l -> b l t c')

    # conjugate x2
    x2_hat = hat_layer(x2.transpose(2, -1))
    conj_x2_hat = torch.matmul(H, torch.matmul(x2_hat, torch.inverse(H)))
    # print(conj_x2_hat.shape)
    conj_x2 = rearrange(vee_sp4(conj_x2_hat), 'b c t l -> b l t c')
    inv_output = torch.zeros((1, num_testing, 1))
    # compute invariant function
    for n in range(num_testing):
        inv_output[0, n, 0] = invariant_function(
            x1_hat[0, 0, n, :, :], x2_hat[0, 0, n, :, :])
        
        # for i in range(num_conjugate):
        #     print(invariant_function(conj_x1_hat[0, i, n, :, :],conj_x2_hat[0, i, n, :, :]))

    test_data['x1'] = x1.numpy().reshape(10, num_testing)
    test_data['x2'] = x2.numpy().reshape(10, num_testing)
    test_data['x1_conjugate'] = conj_x1.numpy().reshape(10, num_testing, num_conjugate)
    test_data['x2_conjugate'] = conj_x2.numpy().reshape(10, num_testing, num_conjugate)
    test_data['y'] = inv_output.numpy().reshape(1, num_testing)

    np.savez(data_saved_path + data_name + "_test_data.npz", **test_data)

    print("Done! Data saved to: \n", data_saved_path +
          data_name + "_train_data.npz\n", data_saved_path + data_name + "_test_data.npz")
