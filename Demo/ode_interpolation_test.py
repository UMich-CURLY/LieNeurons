import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
import datetime

from PlotFunction import Visualizer

parser = argparse.ArgumentParser('ODE demo with input')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

args.viz = True
args.batch_time = 30
# args.batch_size = 20





if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

def init_writer():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"logs/{current_time}"
    writer = SummaryWriter(log_dir)
    writer.add_text("num_iterations: ", str(args.niters))
    return writer


device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

m = 2
true_y0 = torch.tensor([[2., 0.]]).to(device)
t = torch.linspace(0., 25., args.data_size).to(device)
Dt = t[1] - t[0]
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)

B_input = torch.tensor([[1., 1.],[0.0, 1.0]]).to(device)
# print(B_input.shape)

func_u = lambda t: torch.tensor([[torch.sin(t), torch.cos(t)]]).to(device)
true_u0 = func_u(t[0])

interpolation_method = "first_order" # "zero_order" or "first_order"

def interpolate_u(t, u,Dt):
    ii = int(t // Dt)
    match interpolation_method:
        case "zero_order":
            if ii >= len(u) - 1:
                ii = len(u) - 1
            u_t_interpolated = u[ii]
        case "first_order":
            
            if ii >= len(u) - 1:
                ii = len(u) - 1
                u_t_interpolated = u[ii]
            else:
                u_t_interpolated = u[ii] + (t - ii * Dt) * (u[ii + 1] - u[ii]) / Dt
    return u_t_interpolated

class Lambda(nn.Module):
    def forward(self, t, y):
        u_t_interpolated = func_u(t)
        return torch.mm(y**3, true_A) + torch.mm(u_t_interpolated,B_input)
    
    def forward_discrete_u(self, t, y, u):
        u_t_interpolated=interpolate_u(t, u,Dt)
        return torch.matmul(y**3, true_A) + torch.matmul(u_t_interpolated, B_input)
    

with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5').to(device)
    true_u = torch.stack([func_u(t[i]) for i in range(t.shape[0])], dim=0).to(device)



from layer import ODEFunc

func_test = ODEFunc().to(device)
state_dic_all = torch.load("weights/demo/ode_input_interpolation.pth")
func_test.load_state_dict(state_dic_all['model_state_dict'])

iter_best = state_dic_all['iteration']
print("iter_best: ", iter_best)
loss_best = state_dic_all['loss']
print("loss_best: ", loss_best)

func_with_u_val = lambda t,y: func_test(t,y,true_u,Dt).to(device)

with torch.no_grad():
    pred_y_val = odeint(func_with_u_val, true_y0, t).to(device)

Visualizer_trueTra = Visualizer("figures/ode_interpolation/")
Visualizer_trueTra.Plot_XY_Tra(true_y, pred_y_val, t, figurename = "test_after_train")
