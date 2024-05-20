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

# args.viz = True
args.batch_time = 30
# args.batch_size = 20

torch.manual_seed(5566)

interpolation_method = "first_order" # "zero_order" or "first_order"


if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

def init_writer():
    # current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # log_dir = f"logs/{current_time}"
    log_dir = "logs/ode_input_interpolation/"
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

func_u = lambda t: torch.tensor([[torch.sin(t), torch.cos(t)]]).to(device)

def interpolate_u(t, u, Dt):
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

Visualizer_trueTra = Visualizer("figures/ode_interpolation/")
Visualizer_trueTra.Plot_XY_Tra(true_y, true_u, t, figurename = "True_Tra_and_u")

# Test interpolation method
true_y_with_discrete_u = odeint(lambda t,y:Lambda().forward_discrete_u(t,y,true_u), true_y0, t).to(device)
Visualizer_trueTra.Plot_XY_Tra(true_y, true_y_with_discrete_u, t, figurename = "truey_and_y_interpolated")


def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s] # (M, D)
    # batch_t = torch.stack([t[s + i] for i in range(args.batch_time)], dim=0).T
    batch_t = t[:args.batch_time]
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0) 
    batch_u = torch.stack([true_u[s + i] for i in range(args.batch_time)], dim=0)  

    return batch_y0.to(device), batch_t.to(device), batch_y.to(device), batch_u.to(device)


# class ODEFunc(nn.Module):

#     def __init__(self):
#         super(ODEFunc, self).__init__()

#         self.net = nn.Sequential(
#             nn.Linear(4, 128),
#             nn.ReLU(),
#             nn.Linear(128, 2),
#         )

#         for m in self.net.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, mean=0, std=0.1)
#                 nn.init.constant_(m.bias, val=0)

#     def forward(self, t, y: torch.Tensor, u: torch.Tensor, Dt):
#         ii = int(t // Dt)
#         if ii >= len(u) - 1:
#             ii = len(u) - 1
#             u_t_interpolated = u[ii]
#         else:
#             u_t_interpolated = u[ii] + (t - ii * Dt) * (u[ii + 1] - u[ii]) / Dt
#         # print("y.shape: ", y.shape)
#         # print("u.shape: ", u.shape)
#         # print("u_t_interpolated.shape: ", u_t_interpolated.shape)
#         return self.net(torch.cat((y**3, u_t_interpolated), dim=y.dim() - 1))
    
from layer import ODEFunc

# Test
batch_y0, batch_t, batch_y, batch_u = get_batch()
print("batch_y0.shape: ", batch_y0.shape)
print("batch_t.shape: ", batch_t.shape)
print("batch_y.shape: ", batch_y.shape)
print("batch_u.shape: ", batch_u.shape)
odefunc_test = ODEFunc().to(device)
func_test = lambda t,y: odefunc_test(t,y,batch_u,Dt).to(device)
temp_test = odeint(func_test, batch_y0, batch_t).to(device)


if __name__ == '__main__':

    writer = init_writer()
    best_loss = float('inf')

    if args.viz:
        Visualizer_train = Visualizer("figures/ode_interpolation_train/")

    ii = 0

    func = ODEFunc().to(device)
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y, batch_u = get_batch()

        func_with_u = lambda t,y: func(t,y,batch_u,Dt).to(device)
        pred_y = odeint(func_with_u, batch_y0, batch_t).to(device)
        
        loss = torch.mean(torch.abs(pred_y - batch_y))
        
        loss.backward()
        optimizer.step()

        writer.add_scalar("training_loss", loss.item(), itr)


        if itr % args.test_freq == 0:
            with torch.no_grad():
                func_with_u_val = lambda t,y: func(t,y,true_u,Dt).to(device)
                pred_y_val = odeint(func_with_u_val, true_y0, t).to(device)
                loss = torch.mean(torch.abs(pred_y_val - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                writer.add_scalar("Validation_loss", loss.item(), itr)
                if args.viz:
                    Visualizer_train.Plot_XY_Tra( true_y,pred_y_val, t, itr,realtime_draw = False)
                ii += 1

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    state = {'iteration': itr,
                        'model_state_dict': func.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}
                    
                    model_dir = "weights/demo/"
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    torch.save(state, model_dir + "ode_input_interpolation.pth")

        end = time.time()
