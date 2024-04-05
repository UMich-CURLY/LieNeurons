import sys  # nopep8
sys.path.append('.')  # nopep8

import numpy as np
import scipy.linalg
import torch
import math
import time

from core.lie_alg_util import *
from core.lie_group_util import *
from experiment.so3_bch_layers import *

if __name__ == "__main__":


    num_test = 10000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using ', device)

    so3_hatlayer = HatLayer(algebra_type='so3')

    def gen_random_rotation_vector():
        v = torch.rand(1,3)
        v = v/torch.norm(v)
        phi = (math.pi-1e-6)*torch.rand(1)
        v = phi*v
        return v
    sum_t_first_order = 0
    sum_t_second_order = 0
    sum_t_third_order = 0
    sum_t_mlp = 0
    sum_t_LN = 0

    model = SO3EquivariantReluBracketLayers(2).to(device)

    # model = SO3EquivariantReluLayers(2).to(device)
    # model = SO3EquivariantBracketLayers(2).to(device)
    mlp_model = MLP(6).to(device)

    checkpoint = torch.load('/home/justin/code/LieNeurons/weights/0120_so3_bch_relu_bracket_best_test_loss_acc.pt')
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)

    checkpoint = torch.load('/home/justin/code/LieNeurons/weights/0327_so3_bch_mlp_4_layers_1024_augmented_best_test_loss_acc.pt')
    mlp_model.load_state_dict(checkpoint['model_state_dict'],strict=False)

    # generate training data
    for i in range(num_test):
        # generate random v1, v2
        v1 = gen_random_rotation_vector()
        v2 = gen_random_rotation_vector()

        K1 = so3_hatlayer(v1).to(device)
        K2 = so3_hatlayer(v2).to(device)


        t1 = time.time()
        out = BCH_first_order_approx(K1,K2)
        t2 = time.time()
        sum_t_first_order += t2-t1

        t1 = time.time()
        out = BCH_second_order_approx(K1,K2)
        t2 = time.time()
        sum_t_second_order += t2-t1

        t1 = time.time()
        out = BCH_third_order_approx(K1,K2)
        t2 = time.time()
        sum_t_third_order += t2-t1

        v1 = v1.to(device)
        v2 = v2.to(device)
        v = torch.cat((v1,v2),0)

        v = rearrange(v, 'n k -> 1 n k 1')

        t1 = time.time()
        out = model(v)
        t2 = time.time()
        sum_t_LN += t2-t1

        t1 = time.time()
        out = mlp_model(v)
        t2 = time.time()
        sum_t_mlp += t2-t1


    print("First order time: ", sum_t_first_order/num_test)
    print("Second order time: ", sum_t_second_order/num_test)
    print("Third order time: ", sum_t_third_order/num_test)
    print("LN time: ", sum_t_LN/num_test)
    print("MLP time: ", sum_t_mlp/num_test)
        