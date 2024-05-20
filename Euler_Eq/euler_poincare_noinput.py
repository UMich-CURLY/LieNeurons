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

args.num_training = 1
args.viz = True
args.model_type = "LN_ode"
# args.model_type = "ODEMyTest"


if args.model_save_path == 'unspecified':
    args.model_save_path = "weights/Euler_NoInput/Single_Tra/"+args.model_type + "/"
if args.fig_save_path == 'unspecified':
    args.fig_save_path = "figures/Euler_NoInput/Single_Tra/"+args.model_type + "/"
if args.log_writer_path == 'unspecified':
    args.log_writer_path = "logs/Euler_NoInput/Single_Tra/"+args.model_type + "/"

args.batch_time = 30

if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

torch.manual_seed(5566)
if args.num_training >1:
    training_y0 = torch.rand((args.num_training, 3)).to(device)
else:
    training_y0 = torch.tensor([[2., 1.,3.0]]).to(device)
val_true_y0 = torch.tensor([[2., 1.,3.0]]).to(device)
t = torch.linspace(0., 25., args.data_size).to(device)
t_val = torch.linspace(0., 5., int(args.data_size/5)).to(device)

class EulerPoincareEquation(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        '''
        Inertia matrix of the ISS
        https://athena.ecs.csus.edu/~grandajj/ME296M/space.pdf
        page 7-62
        '''
        if args.inertia_type == 'iss':
            self.I = torch.Tensor([[5410880., -246595., 2967671.],[-246595., 29457838., -47804.],[2967671., -47804., 26744180.]]).unsqueeze(0).to(device)
        elif args.inertia_type == 'model1':
            self.I = torch.Tensor([[12, -5., 7.],[-5., 20., -2.],[7., -2., 5.]]).unsqueeze(0).to(device)
        self.I_inv = torch.inverse(self.I)
        self.hat_layer = HatLayer(algebra_type='so3').to(device)

    def forward(self,t,w):
        '''
        w: angular velocity (B,3) or (1,3)
        '''
        w_v = w.unsqueeze(2)
        return -torch.matmul(self.I_inv,torch.matmul(self.hat_layer(w),torch.matmul(self.I,w_v))).squeeze(2)


with torch.no_grad():
    training_y = []
    for i in range(args.num_training):
        true_y = odeint(EulerPoincareEquation(), training_y0[i,:].unsqueeze(0), t, method='dopri5')
        training_y.append(true_y)   

    val_true_y = odeint(EulerPoincareEquation(), val_true_y0, t_val, method='dopri5')

print("training_y", training_y[0].shape)

num_Tra_train = len(training_y)
if args.num_training > num_Tra_train:
    KeyError("args.num_training should be less than the number of training trajectories")

from PlotFunction import Visualizer
true_y = training_y[0]
Visualize_true = Visualizer(args.fig_save_path)
Visualize_true.Plot_XY_Tra(true_y,true_y,t,figurename="trueXY_and_trueXY",label2="True y")

def get_training_batch():
    j = random.randint(0, args.num_training - 1)
    y_j = training_y[j]
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = y_j[s] # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([y_j[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, 3)

    return batch_y0, batch_t, batch_y


def init_writer():
    writer = SummaryWriter(
        args.log_writer_path)
    writer.add_text("num_iterations: ", str(args.niters))
    return writer

if __name__ == '__main__':


    writer = init_writer()
    if args.viz:
        Visualizer_train = Visualizer(args.fig_save_path)

    func = model_choose(args.model_type,device=device)
    # func = 
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()
    
    best_loss = float('inf')

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_training_batch()

        pred_y = odeint(func, batch_y0, batch_t, atol=1e-9,rtol=1e-7).to(device)

        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()

        writer.add_scalar('training loss', loss.item(), itr)
        optimizer.step()

        
        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, val_true_y0.unsqueeze(0), t_val)
        
                pred_y = pred_y[:,0,:,:]
     
                loss = torch.mean(torch.abs(pred_y - val_true_y))
 

                writer.add_scalar('validation loss', loss.item(), itr)
                
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                if args.viz:
                    Visualizer_train.Plot_XY_Tra(val_true_y,pred_y,t_val,iter=itr)

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
