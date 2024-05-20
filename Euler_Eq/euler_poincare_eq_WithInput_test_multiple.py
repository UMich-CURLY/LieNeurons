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
parser.add_argument('--num_testing', type=int, default=2)
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

args.num_testing = 2 # number of trajectories for testing, last args.num_testing trajectories are testing data
args.viz = True
# args.model_type = "neural_ode_WithInput"
# args.model_type = "LN_ode_WithInput1"

print("Training Model: ", args.model_type)

args.fig_save_path = "figures/Euler_WithInput/Multi_Tra/"+args.model_type + "/"
model_dir = "weights/Euler_WithInput/Multi_Tra/"+args.model_type + "/"

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


from PlotFunction import Visualizer

if __name__ == '__main__':
    data_all = torch.load("data/Euler_WithInput/Euler_WithInput_train_muliple_include_test.pt")
    testing_y = data_all['y']  
    testing_u = data_all['u']
    t = data_all['t']
    testing_y0 = data_all['y0'] 
    """
    testing_y: include traning and testing data, let last args.num_testing be testing data 
    same for testing_u, testing_y0
    shpe of testing_y: list of [num_sample_everyTra, 1, 3]

    testing_y0: tensor, tensor_size =  [num_traj, 3]
    """

    Dt = t[1] - t[0]
    if args.viz:
        Visualize_true = Visualizer(args.fig_save_path)
    
    func = model_choose(args.model_type,device=device)
    model_path = model_dir + args.model_type + "_best_val_loss_acc.pt"
    func.load_state_dict(torch.load(model_path)['model_state_dict'])

    ## 
    true_y = torch.stack(testing_y[-args.num_testing:], dim=1) # true_y: time_series * num_testing * 1 * 3
    u_temp = torch.stack(testing_u[-args.num_testing:], dim=1) # u_temp: time_series * num_testing * 1 * 3

    true_y0 = testing_y0[-args.num_testing:] # true_y0: num_testing * 3
    true_y0 = true_y0.unsqueeze(1) # true_y0: num_testing * 1 * 3

    func_with_u_test = lambda t,y: func(t,y,u_temp,Dt)
    with torch.no_grad():
        pred_y = odeint(func_with_u_test, true_y0, t).to(device)

    loss = torch.mean(torch.abs(pred_y - true_y))
    print("Testing Loss : ", loss.item())

    if args.viz:
        for i in range(args.num_testing):
            Visualize_true.Plot_XYZ_Tra(true_y[:,i,:,:], pred_y[:,i,:,:], t, figurename="Trajectory_"+str(i))

    
