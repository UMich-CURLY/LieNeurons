import sys  # nopep8
sys.path.append('.')  # nopep8

import numpy as np
import scipy.linalg
import torch
import math

from core.lie_alg_util import *
from core.lie_group_util import *

if __name__ == "__main__":

    data_saved_path = "data/so3_bch_data/"
    data_name = "so3_bch_10000_augmented"
    gen_augmented_training_data = False

    num_training = 5000
    num_testing = 5000
    num_conjugate = 1

    train_data = {}
    test_data = {}
    so3_hatlayer = HatLayer(algebra_type='so3')

    def gen_random_rotation_vector():
        v = torch.rand(1,3)
        v = v/torch.norm(v)
        phi = (math.pi-1e-6)*torch.rand(1)
        v = phi*v
        return v

    x1 = torch.zeros((num_training,3))
    x2 = torch.zeros((num_training,3))
    y = torch.zeros((num_training,3))

    # if we want to generate adjoint augmented training data
    if(gen_augmented_training_data):
        x1_conj = torch.zeros((num_training, num_conjugate,3))
        x2_conj = torch.zeros((num_training, num_conjugate,3))
        y_conj = torch.zeros((num_training, num_conjugate,3))
        R_aug = torch.zeros((num_training, num_conjugate,3,3))

    # generate training data
    for i in range(num_training):
        # generate random v1, v2
        v1 = gen_random_rotation_vector()
        v2 = gen_random_rotation_vector()

        K1 = so3_hatlayer(v1)
        K2 = so3_hatlayer(v2)

        R1 = exp_so3(K1[0,:,:])
        R2 = exp_so3(K2[0,:,:])

        R3 = torch.matmul(R1,R2)
        
        v3 = vee(log_SO3(R3), algebra_type='so3')

        # v3 = vee(BCH_approx(K1[0,:,:], K2[0,:,:]), algebra_type='so3')
        if(torch.norm(v3) > math.pi):
            print("----------output bigger than pi---------")
            print("norm v1", torch.norm(v1))
            print("norm v2", torch.norm(v2))
            print("norm v3", torch.norm(v3))
            print("v1",v1)
            print("v2",v2)
            print("v3",v3)
            print("R1",R1)
            print("R2",R2)
            print("R3",R3)

        x1[i,:] = v1
        x2[i,:] = v2
        y[i,:] = v3

        if(gen_augmented_training_data):
            for j in range(num_conjugate):
                v4 = gen_random_rotation_vector()
                K4 = so3_hatlayer(v4)
                R4 = exp_so3(K4[0,:,:])

                R1_conj = R4@R1@R4.T
                R2_conj = R4@R2@R4.T
                R3_conj = R1_conj@R2_conj

                v1_conj = vee(log_SO3(R1_conj), algebra_type='so3')
                v2_conj = vee(log_SO3(R2_conj), algebra_type='so3')
                v3_conj = vee(log_SO3(R3_conj), algebra_type='so3')

                x1_conj[i,j,:] = v1_conj
                x2_conj[i,j,:] = v2_conj
                y_conj[i,j,:] = v3_conj
                R_aug[i,j,:,:] = R4

    if(gen_augmented_training_data):
        # concatenate the conjugate data
        train_data['x1'] = torch.cat((x1,x1_conj.reshape(num_training*num_conjugate,3)),dim=0).numpy()
        train_data['x2'] = torch.cat((x2,x2_conj.reshape(num_training*num_conjugate,3)),dim=0).numpy()
        train_data['y'] = torch.cat((y,y_conj.reshape(num_training*num_conjugate,3)),dim=0).numpy()
        train_data['R_aug'] = R_aug.numpy()
    else:
        train_data['x1'] = x1.numpy()
        train_data['x2'] = x2.numpy()
        train_data['y'] = y.numpy()

    np.savez(data_saved_path + data_name + "_train_data.npz", **train_data)



    # generate testing data
    x1 = torch.zeros((num_testing,3))
    x2 = torch.zeros((num_testing,3))
    y = torch.zeros((num_testing,3))

    x1_conj = torch.zeros((num_testing, num_conjugate,3))
    x2_conj = torch.zeros((num_testing, num_conjugate,3))
    y_conj = torch.zeros((num_testing, num_conjugate,3))
    R_aug = torch.zeros((num_testing, num_conjugate,3,3))

    for i in range(num_testing):
        # generate random v1, v2
        v1 = gen_random_rotation_vector()
        v2 = gen_random_rotation_vector()

        K1 = so3_hatlayer(v1)
        K2 = so3_hatlayer(v2)

        R1 = exp_so3(K1[0,:,:])
        R2 = exp_so3(K2[0,:,:])

        R3 = torch.matmul(R1,R2)
        v3 = vee(log_SO3(R3), algebra_type='so3')
        
        # v3 = vee(BCH_approx(K1[0,:,:], K2[0,:,:]), algebra_type='so3')

        x1[i,:] = v1
        x2[i,:] = v2
        y[i,:] = v3

        for j in range(num_conjugate):
            v4 = gen_random_rotation_vector()
            K4 = so3_hatlayer(v4)
            R4 = exp_so3(K4[0,:,:])

            R1_conj = R4@R1@R4.T
            R2_conj = R4@R2@R4.T
            R3_conj = R1_conj@R2_conj

            v1_conj = vee(log_SO3(R1_conj), algebra_type='so3')
            v2_conj = vee(log_SO3(R2_conj), algebra_type='so3')
            v3_conj = vee(log_SO3(R3_conj), algebra_type='so3')

            x1_conj[i,j,:] = v1_conj
            x2_conj[i,j,:] = v2_conj
            y_conj[i,j,:] = v3_conj
            R_aug[i,j,:,:] = R4

    test_data['x1'] = x1.numpy()
    test_data['x2'] = x2.numpy()
    test_data['x1_conjugate'] = x1_conj.numpy()
    test_data['x2_conjugate'] = x2_conj.numpy()
    test_data['y'] = y.numpy()
    test_data['y_conj'] = y_conj.numpy()
    test_data['R_aug'] = R_aug.numpy()

    np.savez(data_saved_path + data_name + "_test_data.npz", **test_data)

    print("Done! Data saved to: \n", data_saved_path +
          data_name + "_train_data.npz\n", data_saved_path + data_name + "_test_data.npz")