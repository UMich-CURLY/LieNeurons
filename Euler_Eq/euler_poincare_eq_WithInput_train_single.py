import sys  # nopep8
sys.path.append('.')  # nopep8

import os
import argparse
import time
import numpy as np

import time
import random

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
parser.add_argument('--save_freq', type=int, default=100) # save model every 100 iterations
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--inertia_type', type=str, choices=['iss','model1'], default='iss')
parser.add_argument('--model_type', type=str, default='neural_ode')
parser.add_argument('--model_save_path', type=str, default='unspecified')
parser.add_argument('--fig_save_path', type=str, default='unspecified')
parser.add_argument('--log_writer_path', type=str, default='unspecified')
args = parser.parse_args()

args.num_training = 1 # do not change this, this is for single trajectory training

# args.viz = True
# args.model_type = "neural_ode_WithInput"
# args.model_type = "LN_ode_WithInput"
# args.model_type = "LN_ode_WithInput1"
# args.model_type = "LN_ode_WithInput6"

print("Training Model: ", args.model_type)

if args.model_save_path == 'unspecified':
    args.model_save_path = "weights/Euler_WithInput/Single_Tra/"+args.model_type + "/"
if args.fig_save_path == 'unspecified':
    args.fig_save_path = "figures/Euler_WithInput/Single_Tra/"+args.model_type + "/"
if args.log_writer_path == 'unspecified':
    args.log_writer_path = "logs/Euler_WithInput/Single_Tra/"+args.model_type + "/"

args.batch_time = 30

if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


data_train = torch.load("data/Euler_WithInput/Euler_WithInput_train_muliple_include_test.pt")
training_y = data_train['y']  # list num = num_traj,  tensor_size =  [num_sample, 1, 3]
training_u = data_train['u']  # list num = num_traj,  tensor_size =  [num_sample, 1, 3]
t = data_train['t'] # tensor_size =  [num_sample]
training_y0 = data_train['y0'] # tensor_size =  [num_traj, 3]

Dt = t[1] - t[0]

print("training_y", training_y[0].shape)
print("training_u", training_u[0].shape)

num_Tra_train = len(training_y)
if args.num_training > num_Tra_train:
    KeyError("args.num_training should be less than the number of training trajectories")

from PlotFunction import Visualizer
true_y = training_y[0]
true_u = training_u[0]
Visualize_true = Visualizer(args.fig_save_path)
Visualize_true.Plot_XY_Tra(true_y,true_u,t,figurename="trueXY_and_u12",label2="True u")
Visualize_true.Plot_XYZ_Tra(true_y,true_u,t,figurename="trueXYX_and_u123", label2="True u")

def get_training_batch():
    j = random.randint(0, args.num_training - 1)
    y_j = training_y[j]
    u_j = training_u[j]
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = y_j[s] # (M, D)

    batch_t = t[:args.batch_time]
    batch_y = torch.stack([y_j[s + i] for i in range(args.batch_time)], dim=0) 
    batch_u = torch.stack([u_j[s + i] for i in range(args.batch_time)], dim=0)  

    return batch_y0.to(device), batch_t.to(device), batch_y.to(device), batch_u.to(device)

## TEST
# funcc = LNODEFunc8(device=device).to(device)
# x = torch.rand((1,1,3)).to(device)
# output = funcc(0.,x)
# print("output.shape", output.shape)

def init_writer():
    writer = SummaryWriter(
        args.log_writer_path)
    writer.add_text("num_iterations: ", str(args.niters))
    return writer



if __name__ == '__main__':


    writer = init_writer()
    if args.viz:
        Visualizer_train = Visualizer(args.fig_save_path)

    func = model_choose(args.model_type, device=device)
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()
    
    if os.path.exists(args.model_save_path + args.model_type + '_best_val_loss_acc.pt'):
        best_loss = torch.load(args.model_save_path + args.model_type + '_best_val_loss_acc.pt')['loss']
        print("model exists, best loss: ", best_loss)
    else:
        best_loss = float('inf')
        print("model does not exist, initialize best loss")

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y, batch_u = get_training_batch() 
        """
        batch_y0: num_traj * 1 * 3
        batch_t: time_series
        batch_y: time_series * num_traj * 1 * 3 
        batch_u: time_series * num_traj * 1 * 3 
        """
        func_with_u = lambda t,y: func(t,y,batch_u,Dt)

        pred_y = odeint(func_with_u, batch_y0, batch_t, atol=1e-9,rtol=1e-7).to(device)

        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()

        writer.add_scalar('training loss', loss.item(), itr)
        optimizer.step()

        
        if itr % args.test_freq == 0:
            with torch.no_grad():
                u_temp = training_u[0] # u_temp: time_series * 1 * 3
                u_temp = u_temp.unsqueeze(1) # u_temp: time_series * 1 * 1 * 3
                y0_temp = training_y0[0,:].unsqueeze(0).unsqueeze(0) # y0_temp: 1 * 1 * 3
                func_with_u_val = lambda t,y: func(t,y,u_temp,Dt)
                pred_y = odeint(func_with_u_val, y0_temp, t) # pred_y: time_series 1 * 1 * 3
                pred_y.squeeze_(1) # pred_y: time_series * 1 * 3
                loss = torch.mean(torch.abs(pred_y - training_y[0]))

                writer.add_scalar('validation loss', loss.item(), itr)
                
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                if args.viz:
                    Visualizer_train.Plot_XY_Tra(training_y[0],pred_y,t,iter=itr)

            if loss < best_loss:
                best_loss = loss

                state = {'iteration': itr,
                        'model_state_dict': func.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}

                torch.save(state, args.model_save_path + args.model_type +
                        '_best_val_loss_acc.pt')
            print("------------------------------")

        if itr % args.save_freq == 0:
            state = {'iteration': itr,
                        'model_state_dict': func.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}

            torch.save(state, args.model_save_path + args.model_type + '_iter_'+str(itr)+'.pt')
            
        end = time.time()
