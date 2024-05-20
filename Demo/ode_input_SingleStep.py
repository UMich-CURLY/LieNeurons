import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser('ODE demo')
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

args.data_size = 1000
args.viz = True
# args.adjoint = True
args.batch_size = 1
args.batch_time = 50

torch.manual_seed(5566)

def init_writer():
    writer = SummaryWriter("logs/ode_input_SingleStep/")
    writer.add_text("num_iterations: ", str(args.niters))
    return writer

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([[2., 0.]]).to(device)
t = torch.linspace(0., 25., args.data_size).to(device)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)

B_input = torch.tensor([[1., 1.],[0.0, 1.0]]).to(device)
# print(B_input.shape)
print("t.shape: ", t.shape)

func_u = lambda t: torch.tensor([[torch.sin(t), torch.cos(t)]]).to(device)

class Lambda(nn.Module):

    def forward(self, t, y):
        ut = func_u(t)
        return torch.mm(y**3, true_A)  + ut @ B_input
    def forward_u(self, t, y, u):
        return torch.mm(y**3, true_A) + u @ B_input
    
N = args.data_size

sin_t = torch.sin(t)
cos_t = torch.cos(t)
# u_all = torch.stack((sin_t, cos_t), dim=1).to(device)
u_all = torch.stack([func_u(t[i]) for i in range(t.shape[0])], dim=0).to(device)
true_y = torch.zeros(N, 2).to(device)

# u_all = 2.2 * torch.ones(N, 1).to(device)
# print("u_all shape: ", u_all.shape)

def odeint_with_input(func, y0: torch.Tensor, t: torch.Tensor, u: torch.Tensor):
    # true_y0.shape = [1,2]
    # t.shape = [1000]
    # u.shape = [1000,1]
    N = t.shape[0]
    for i in range(N-1):
        temp = odeint(lambda t,y: func(t,y,u[i,:]), y0, t[i:2+i], method='dopri5')
        if i == 0:
            pred_y = temp
        else:
            pred_y = torch.cat((pred_y, temp[1,:,:].unsqueeze(0)), dim=0)
        y0 = temp[1,:,:]
    return pred_y

with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t)
    true_y_zero = odeint_with_input(Lambda().forward_u, true_y0, t, u_all)

# TEST
from PlotFunction import Visualizer
Visualizer_trueTra = Visualizer("figures/ode_input_singleStep/")
Visualizer_trueTra.Plot_XY_Tra(true_y, true_y_zero, t, figurename = "truey_and_truey_zero")


def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s] # (M, D)
    batch_t = t[s:s + args.batch_time]  
    
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, 3)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device),s


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y, u):
        u = u.unsqueeze(0)
        # u = u.unsqueeze(0)
        return self.net(torch.cat((y**3, u), dim=2))
    



if __name__ == '__main__':

    writer = init_writer()
    best_loss = float('inf')

    if args.viz:
        Visualizer_train = Visualizer("figures/ode_input_singleStep_train/")

    ii = 0

    func = ODEFunc().to(device)
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y,s = get_batch()
        batch_length = batch_t.shape[0]
        # print("batch_y0.shape: ", batch_y0.shape)
        # print("batch_t.shape: ", batch_t.shape)
        # print("batch_y.shape: ", batch_y.shape)
        
        pred_y = odeint_with_input(func, batch_y0, batch_t, u_all[s:s+batch_length])
        # print("pred_y.shape: ", pred_y.shape)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        writer.add_scalar("training_loss", loss.item(), itr)


        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint_with_input(func, true_y0.unsqueeze(0), t, u_all)
                pred_y = pred_y.squeeze(1)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                writer.add_scalar("Validation_loss", loss.item(), itr)
                if args.viz:
                    Visualizer_train.Plot_XY_Tra(true_y, pred_y, t,itr)
                ii += 1

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    state = {'iteration': iter,
                        'model_state_dict': func.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}
                    
                    torch.save(state, "models/ode_input.pth")

        end = time.time()
