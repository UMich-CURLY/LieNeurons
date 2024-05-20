import sys  # nopep8
sys.path.append('.')  # nopep8

import os
import argparse
import time
import numpy as np

import time


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from core.lie_neurons_layers import *
from core.lie_alg_util import *
from euler_poincare_eq_layers import *


parser = argparse.ArgumentParser('Euler Poincare Equation Fitting')
parser.add_argument('--method', type=str, default='dopri5')
parser.add_argument('--num_training', type=int, default=10)
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--save_freq', type=int, default=100)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--inertia_type', type=str, choices=['iss','model1'], default='iss')
parser.add_argument('--fig_save_path', type=str, default='figures/euler_poincare_ln')
parser.add_argument('--model_save_path', type=str, default='weights/euler_poincare_ln')
parser.add_argument('--model_type', type=str, default='neural_ode')
parser.add_argument('--log_writer_path', type=str, default='logs/euler_WithInput_ln')
args = parser.parse_args()

# args.viz = True
# args.model_type = "neural_ode_WithInput"
# args.model_type = "LN_ode_WithInput1"

print("Training Model: ", args.model_type)

args.fig_save_path = "figures/Euler_WithInput/Single_Tra/"+args.model_type + "/"
model_dir = "weights/Euler_WithInput/Single_Tra/"+args.model_type + "/"

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


from PlotFunction import Visualizer

if __name__ == '__main__':
    data_test = torch.load("data/Euler_WithInput/Euler_WithInput_test_single.pt")
    testing_y = data_test['y']
    testing_u = data_test['u']
    t = data_test['t']
    testing_y0 = data_test['y0']

    Dt = t[1] - t[0]

    Visualize_true = Visualizer(args.fig_save_path)
    
    func = model_choose(args.model_type,device=device)
    model_path = model_dir + args.model_type + "_best_val_loss_acc.pt"
    func.load_state_dict(torch.load(model_path)['model_state_dict'])

    ## Case 1 : Same as the training data
    true_y = testing_y[0]
    true_y0 = testing_y0[0,:].unsqueeze(0).unsqueeze(0) # shape: [1, 1, 3]
    true_u = testing_u[0].unsqueeze(1) # time_series * 1 * 1 * 3 

    func_with_input = lambda t,y: func(t,y,true_u,Dt)
    with torch.no_grad():
        pred_y = odeint(func_with_input, true_y0, t).to(device)
    pred_y.squeeze_(1)
    loss = torch.mean(torch.abs(pred_y - true_y))
    print("Case 1 : Same as the training data --- Loss : ", loss.item())
    Visualize_true.Plot_XYZ_Tra(true_y, pred_y, t, figurename="Same_y0_ut")

    ## Case 2 : Different y0 same ut
    true_y = testing_y[1]
    true_y0 = testing_y0[1,:].unsqueeze(0).unsqueeze(0) # shape: [1, 1, 3]
    true_u = testing_u[1].unsqueeze(1) # time_series * 1 * 1 * 3 

    func_with_input = lambda t,y: func(t,y,true_u,Dt)
    with torch.no_grad():
        pred_y = odeint(func_with_input, true_y0, t).to(device)
    pred_y.squeeze_(1)
    loss = torch.mean(torch.abs(pred_y - true_y))
    print("Case 2 : Different y0 same ut --- Loss : ", loss.item())
    Visualize_true.Plot_XYZ_Tra(true_y, pred_y, t, figurename="Different_y0_same_ut")

    ## Case 3 : Same y0 different ut
    true_y = testing_y[2]
    true_y0 = testing_y0[2,:].unsqueeze(0).unsqueeze(0) # shape: [1, 1, 3]
    true_u = testing_u[2].unsqueeze(1) # time_series * 1 * 1 * 3

    func_with_input = lambda t,y: func(t,y,true_u,Dt)
    with torch.no_grad():
        pred_y = odeint(func_with_input, true_y0, t).to(device)
    pred_y.squeeze_(1)
    loss = torch.mean(torch.abs(pred_y - true_y))
    print("Case 3 : Same y0 different ut --- Loss : ", loss.item())
    Visualize_true.Plot_XYZ_Tra(true_y, pred_y, t, figurename="Same_y0_different_ut")

    ## Case 4 : Different y0 different ut
    true_y = testing_y[3]
    true_y0 = testing_y0[3,:].unsqueeze(0).unsqueeze(0) # shape: [1, 1, 3]
    true_u = testing_u[3].unsqueeze(1) # time_series * 1 * 1 * 3

    func_with_input = lambda t,y: func(t,y,true_u,Dt)
    with torch.no_grad():
        pred_y = odeint(func_with_input, true_y0, t).to(device)
    pred_y.squeeze_(1)
    loss = torch.mean(torch.abs(pred_y - true_y))
    print("Case 4 : Different y0 different ut --- Loss : ", loss.item())
    Visualize_true.Plot_XYZ_Tra(true_y, pred_y, t, figurename="Different_y0_different_ut")
    
    ## Case 5 : R * y0, R * ut
    true_y = testing_y[4]
    true_y0 = testing_y0[4,:].unsqueeze(0).unsqueeze(0) # shape: [1, 1, 3]
    true_u = testing_u[4].unsqueeze(1) # time_series * 1 * 1 * 3

    func_with_input = lambda t,y: func(t,y,true_u,Dt)
    with torch.no_grad():
        pred_y = odeint(func_with_input, true_y0, t).to(device)
    pred_y.squeeze_(1)
    loss = torch.mean(torch.abs(pred_y - true_y))
    print("Case 5 : R * y0, R * ut --- Loss : ", loss.item())
    Visualize_true.Plot_XYZ_Tra(true_y, pred_y, t, figurename="R_y0_R_ut")






    # import matplotlib.pyplot as plt
    # plt.waitforbuttonpress()