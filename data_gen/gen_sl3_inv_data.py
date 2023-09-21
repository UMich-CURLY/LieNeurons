import sys  # nopep8
sys.path.append('.')  # nopep8

import numpy as np
import torch
from scipy.linalg import expm

from core.lie_group_util import *
from core.lie_neurons_layers import *

def invariant_function(x1,x2):
    return torch.sin(torch.trace(x1@x1))-torch.pow(torch.trace(x2@x2),3)/2.0+torch.det(x1@x2)


if __name__ == "__main__":
    data_saved_path = "data/sl3_inv_data/"
    num_training = 100
    num_testing = 30
    rnd_scale = 1

    train_data = {}
    test_data = {}

    hat_layer = HatLayerSl3()

    x1 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 8, num_training)))\
                    .reshape(1, 1, 8, num_training)
    x2 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 8, num_training)))\
                    .reshape(1, 1, 8, num_training)
    
    # conjugate transformation
    h = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (num_training,8)))
    H = torch.linalg.matrix_exp(hat_layer(h))
    # conjugate x1
    x1_hat = hat_layer(x1.transpose(2, -1))
    conj_x1_hat = torch.matmul(H, torch.matmul(x1_hat, torch.inverse(H)))
    conj_x1 = vee_sl3(conj_x1_hat).transpose(2, -1)
    # conjugate x2
    x2_hat = hat_layer(x2.transpose(2, -1))
    conj_x2_hat = torch.matmul(H, torch.matmul(x2_hat, torch.inverse(H)))
    conj_x2 = vee_sl3(conj_x2_hat).transpose(2, -1)

    inv_output = torch.zeros((1,num_training,1))
    # compute invariant function
    for n in range(num_training):
        inv_output[0,n,0] = invariant_function(x1_hat[0,0,n,:,:],x2_hat[0,0,n,:,:])

        print("--------------invariant function output: ------------------")
        print(invariant_function(x1_hat[0,0,n,:,:],x2_hat[0,0,n,:,:]))
        print(invariant_function(conj_x1_hat[0,0,n,:,:],conj_x2_hat[0,0,n,:,:]))

    train_data['x1'] = x1.numpy().reshape(8,num_training)
    train_data['x2'] = x2.numpy().reshape(8,num_training)
    train_data['x1_conjugate'] = conj_x1.numpy().reshape(8,num_training)
    train_data['x2_conjugate'] = conj_x2.numpy().reshape(8,num_training)
    train_data['y'] = inv_output.numpy().reshape(1,num_training)

    np.savez(data_saved_path + "sl3_inv_train_data.npz", **train_data)


    x1 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 8, num_testing)))\
                    .reshape(1, 1, 8, num_testing)
    x2 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 8, num_testing)))\
                    .reshape(1, 1, 8, num_testing)
    
    # conjugate transformation
    h = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (num_testing,8)))
    H = torch.linalg.matrix_exp(hat_layer(h))
    # conjugate x1
    x1_hat = hat_layer(x1.transpose(2, -1))
    conj_x1_hat = torch.matmul(H, torch.matmul(x1_hat, torch.inverse(H)))
    conj_x1 = vee_sl3(conj_x1_hat).transpose(2, -1)
    # conjugate x2
    x2_hat = hat_layer(x2.transpose(2, -1))
    conj_x2_hat = torch.matmul(H, torch.matmul(x2_hat, torch.inverse(H)))
    conj_x2 = vee_sl3(conj_x2_hat).transpose(2, -1)

    inv_output = torch.zeros((1,num_testing,1))
    # compute invariant function
    for n in range(num_testing):
        inv_output[0,n,0] = invariant_function(x1_hat[0,0,n,:,:],x2_hat[0,0,n,:,:])


    test_data['x1'] = x1.numpy().reshape(8,num_testing)
    test_data['x2'] = x2.numpy().reshape(8,num_testing)
    test_data['x1_conjugate'] = conj_x1.numpy().reshape(8,num_testing)
    test_data['x2_conjugate'] = conj_x2.numpy().reshape(8,num_testing)
    test_data['y'] = inv_output.numpy().reshape(1,num_testing)

    np.savez(data_saved_path + "sl3_inv_test_data.npz", **test_data)

    print("Done! Data saved to: \n",data_saved_path + "sl3_inv_train_data.npz\n"\
            ,data_saved_path + "sl3_inv_test_data.npz")