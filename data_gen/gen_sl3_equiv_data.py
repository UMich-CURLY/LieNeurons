import sys  # nopep8
sys.path.append('.')  # nopep8

import numpy as np
import torch
from scipy.linalg import expm


from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from core.lie_neurons_layers import *


# def equivariant_function(x1, x2, x3, x4, x5):
#     # return x1@x2@x3@x4@x5 + x1@x1@x1@x1 + x5@x4@x3 + x3@x5@x1 + x1@x2 + x3@x4 + x5@x2
#     return x1@x2@torch.linalg.matrix_exp(x3)@x4@x5 + x1@torch.linalg.matrix_exp(x1)@x1@x1 \
#         + x5@x4@x3 + torch.linalg.matrix_exp(x3)@x5@x1 + x1@x2 + x3@x4 + x5@x2

def sl3_killing_form(x, y):
    return 6*torch.trace(x@y)


def lie_bracket(x, y):
    return x@y - y@x


def equivariant_function(x, y):
    # return sl3_killing_form(x,z)*lie_bracket(lie_bracket(x,y),z) + sl3_killing_form(y,z)*lie_bracket(z,y)
    # return sl3_killing_form(x,z)*lie_bracket(lie_bracket(x,y),z) + sl3_killing_form(y,z)*lie_bracket(z,y)
    # return x@y-y@x
    return lie_bracket(lie_bracket(x, y), y) + lie_bracket(y, x)
    # return 1/4.0*lie_bracket(lie_bracket(x,y),z)


if __name__ == "__main__":
    data_saved_path = "data/sl3_equiv_lie_bracket_data/"
    data_name = "sl3_equiv_10000_lie_bracket_2inputs_augmented"
    gen_augmented_training_data = True
    num_training = 5000
    num_testing = 5000
    num_conjugate = 1
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
    # x4 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 8, num_training)))\
    #     .reshape(1, 1, 8, num_training)
    # x5 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 8, num_training)))\
    #     .reshape(1, 1, 8, num_training)

    # conjugate transformation
    h = torch.Tensor(np.random.uniform(-rnd_scale,
                     rnd_scale, (num_conjugate, num_training, 8)))
    H = torch.linalg.matrix_exp(hat_layer(h))

    # conjugate x1
    x1_hat = hat_layer(x1.transpose(2, -1))
    conj_x1_hat = torch.matmul(H, torch.matmul(x1_hat, torch.inverse(H)))
    conj_x1 = rearrange(vee_sl3(conj_x1_hat), 'b c t l -> b l t c')     # batch, 8, num_training, conjugate

    # conjugate x2
    x2_hat = hat_layer(x2.transpose(2, -1))
    conj_x2_hat = torch.matmul(H, torch.matmul(x2_hat, torch.inverse(H)))
    conj_x2 = rearrange(vee_sl3(conj_x2_hat), 'b c t l -> b l t c')

    # conjugate x3
    # x3_hat = hat_layer(x3.transpose(2, -1))
    # conj_x3_hat = torch.matmul(H, torch.matmul(x3_hat, torch.inverse(H)))
    # conj_x3 = rearrange(vee_sl3(conj_x3_hat), 'b c t l -> b l t c')

    # # conjugate x4
    # x4_hat = hat_layer(x4.transpose(2, -1))
    # conj_x4_hat = torch.matmul(H, torch.matmul(x4_hat, torch.inverse(H)))
    # conj_x4 = rearrange(vee_sl3(conj_x4_hat), 'b c t l -> b l t c')

    # # conjugate x5
    # x5_hat = hat_layer(x5.transpose(2, -1))
    # conj_x5_hat = torch.matmul(H, torch.matmul(x5_hat, torch.inverse(H)))
    # conj_x5 = rearrange(vee_sl3(conj_x5_hat), 'b c t l -> b l t c')

    equ_output = torch.zeros((1, num_training, 8))
    equ_output_conj = torch.zeros((num_conjugate, num_training, 8))
    # compute invariant function
    for n in range(num_training):
        # out = equivariant_function(
        #     x1_hat[0, 0, n, :, :], x2_hat[0, 0, n, :, :],x3_hat[0, 0, n, :, :],\
        #         x4_hat[0, 0, n, :, :],x5_hat[0, 0, n, :, :])
        out = equivariant_function(
            x1_hat[0, 0, n, :, :], x2_hat[0, 0, n, :, :])

        out_vec = vee_sl3(out)

        equ_output[0, n, :] = out_vec

        # print("-------------------------------")
        # print(out)
        for i in range(num_conjugate):
            # conj_out = equivariant_function(conj_x1_hat[0, i, n, :, :],conj_x2_hat[0, i, n, :, :],\
            #                          conj_x3_hat[0, i, n, :, :],conj_x4_hat[0, i, n, :, :],\
            #                          conj_x5_hat[0, i, n, :, :])
            conj_out = equivariant_function(
                conj_x1_hat[0, i, n, :, :], conj_x2_hat[0, i, n, :, :])
            H_i = H[i, n, :, :]
            out_then_conj = torch.matmul(
                H_i, torch.matmul(out, torch.inverse(H_i)))

            equ_output_conj[i, n, :] = vee_sl3(conj_out)

            # print(conj_out - out_then_conj)

            test_result = torch.allclose(
                conj_out, out_then_conj, rtol=1e-4, atol=1e-4)
            # print(test_result)
            if (not test_result):
                print("This is not equivariant")
    if(gen_augmented_training_data):
        train_data['x1'] = torch.cat((x1.reshape(8, num_training),conj_x1.reshape(8, num_training*num_conjugate)),dim=1).numpy()
        train_data['x2'] = torch.cat((x2.reshape(8, num_training),conj_x2.reshape(8, num_training*num_conjugate)),dim=1).numpy()
        train_data['x1_conjugate'] = conj_x1.reshape(8, num_training, num_conjugate).repeat(1,num_conjugate+1,1).numpy()
        train_data['x2_conjugate'] = conj_x2.reshape(8, num_training, num_conjugate).repeat(1,num_conjugate+1,1).numpy()
        train_data['y'] = torch.cat((equ_output.reshape(num_training, 8),equ_output_conj.reshape(num_training*num_conjugate, 8)),dim=0).numpy().reshape(1, num_training*(num_conjugate+1), 8)
        train_data['y_conj'] = equ_output_conj.reshape(
            num_conjugate, num_training, 8).repeat(1,num_conjugate+1,1).numpy()
        train_data['H'] = H.reshape(num_conjugate, num_training, 3, 3).repeat(1,num_conjugate+1,1,1).numpy()
    else:
        train_data['x1'] = x1.numpy().reshape(8, num_training)
        train_data['x2'] = x2.numpy().reshape(8, num_training)
        train_data['x1_conjugate'] = conj_x1.numpy().reshape(8, num_training, num_conjugate)
        train_data['x2_conjugate'] = conj_x2.numpy().reshape(8, num_training, num_conjugate)
        
        train_data['y'] = equ_output.numpy().reshape(1, num_training, 8)
        train_data['y_conj'] = equ_output_conj.numpy().reshape(
            num_conjugate, num_training, 8)
        train_data['H'] = H.numpy().reshape(num_conjugate, num_training, 3, 3)
    # train_data['x1'] = x1.numpy().reshape(8, num_training)
    # train_data['x2'] = x2.numpy().reshape(8, num_training)
    # train_data['x1_conjugate'] = conj_x1.numpy().reshape(
    #     8, num_training, num_conjugate)
    # train_data['x2_conjugate'] = conj_x2.numpy().reshape(
    #     8, num_training, num_conjugate)

    np.savez(data_saved_path + data_name + "_train_data.npz", **train_data)

    '''
    Generate testing data
    '''
    x1 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 8, num_testing)))\
        .reshape(1, 1, 8, num_testing)
    x2 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 8, num_testing)))\
        .reshape(1, 1, 8, num_testing)
    # x3 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 8, num_testing)))\
    #     .reshape(1, 1, 8, num_testing)
    # x4 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 8, num_testing)))\
    #     .reshape(1, 1, 8, num_testing)
    # x5 = torch.Tensor(np.random.uniform(-rnd_scale, rnd_scale, (1, 8, num_testing)))\
    #     .reshape(1, 1, 8, num_testing)

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
    # x3_hat = hat_layer(x3.transpose(2, -1))
    # conj_x3_hat = torch.matmul(H, torch.matmul(x3_hat, torch.inverse(H)))
    # conj_x3 = rearrange(vee_sl3(conj_x3_hat), 'b c t l -> b l t c')

    # # conjugate x4
    # x4_hat = hat_layer(x4.transpose(2, -1))
    # conj_x4_hat = torch.matmul(H, torch.matmul(x4_hat, torch.inverse(H)))
    # conj_x4 = rearrange(vee_sl3(conj_x4_hat), 'b c t l -> b l t c')

    # # conjugate x5
    # x5_hat = hat_layer(x5.transpose(2, -1))
    # conj_x5_hat = torch.matmul(H, torch.matmul(x5_hat, torch.inverse(H)))
    # conj_x5 = rearrange(vee_sl3(conj_x5_hat), 'b c t l -> b l t c')

    equ_output = torch.zeros((1, num_testing, 8))
    equ_output_conj = torch.zeros((num_conjugate, num_testing, 8))
    # compute invariant function
    for n in range(num_testing):
        # out = equivariant_function(
        #     x1_hat[0, 0, n, :, :], x2_hat[0, 0, n, :, :],x3_hat[0, 0, n, :, :],\
        #         x4_hat[0, 0, n, :, :],x5_hat[0, 0, n, :, :])
        out = equivariant_function(
            x1_hat[0, 0, n, :, :], x2_hat[0, 0, n, :, :])
        out_vec = vee_sl3(out)

        equ_output[0, n, :] = out_vec

        # print("-------------------------------")
        # print(out)
        for i in range(num_conjugate):
            # conj_out = equivariant_function(conj_x1_hat[0, i, n, :, :],conj_x2_hat[0, i, n, :, :],\
            #                          conj_x3_hat[0, i, n, :, :],conj_x4_hat[0, i, n, :, :],\
            #                          conj_x5_hat[0, i, n, :, :])
            conj_out = equivariant_function(
                conj_x1_hat[0, i, n, :, :], conj_x2_hat[0, i, n, :, :])
            H_i = H[i, n, :, :]
            out_then_conj = torch.matmul(
                H_i, torch.matmul(out, torch.inverse(H_i)))
            equ_output_conj[i, n, :] = vee_sl3(conj_out)

            # print(conj_out - out_then_conj)

            test_result = torch.allclose(
                conj_out, out_then_conj, rtol=1e-4, atol=1e-4)
            # print(test_result)
            if (not test_result):
                print("This is not equivariant")

        # print("-------------------------------")
        # print(inv_output[0,n,0])
        # for i in range(num_conjugate):
        #     print(invariant_function(conj_x1_hat[0, i, n, :, :],conj_x2_hat[0, i, n, :, :],\
        #                              conj_x3_hat[0, i, n, :, :],conj_x4_hat[0, i, n, :, :],\
        #                              conj_x5_hat[0, i, n, :, :]))

    test_data['x1'] = x1.numpy().reshape(8, num_testing)
    test_data['x2'] = x2.numpy().reshape(8, num_testing)
    # test_data['x3'] = x3.numpy().reshape(8, num_testing)
    # test_data['x4'] = x4.numpy().reshape(8, num_testing)
    # test_data['x5'] = x5.numpy().reshape(8, num_testing)
    test_data['x1_conjugate'] = conj_x1.numpy().reshape(
        8, num_testing, num_conjugate)
    test_data['x2_conjugate'] = conj_x2.numpy().reshape(
        8, num_testing, num_conjugate)
    # test_data['x3_conjugate'] = conj_x3.numpy().reshape(
    #     8, num_testing, num_conjugate)
    # test_data['x4_conjugate'] = conj_x4.numpy().reshape(8, num_testing, num_conjugate)
    # test_data['x5_conjugate'] = conj_x5.numpy().reshape(8, num_testing, num_conjugate)
    test_data['y'] = equ_output.numpy().reshape(1, num_testing, 8)
    test_data['y_conj'] = equ_output_conj.numpy().reshape(
        num_conjugate, num_testing, 8)
    test_data['H'] = H.numpy().reshape(num_conjugate, num_testing, 3, 3)

    np.savez(data_saved_path + data_name + "_test_data.npz", **test_data)

    print("Done! Data saved to: \n", data_saved_path +
          data_name + "_train_data.npz\n", data_saved_path + data_name + "_test_data.npz")
