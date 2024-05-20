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

from PlotFunction import Plot_XY_Tra, visualize

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
args.batch_size = 20

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
print("t.shape: ", t.shape)

func_u = lambda t: torch.tensor([[torch.sin(t), torch.cos(t)]]).to(device)
true_u0 = func_u(t[0])

class Lambda(nn.Module):
    def forward(self, t, y, u=None):
        u_t_interpolated = func_u(t)
        return torch.mm(y**3, true_A) + u_t_interpolated**3 @ B_input
    

with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5').to(device)
    true_u = torch.stack([func_u(t[i]) for i in range(t.shape[0])], dim=0).to(device)

Plot_XY_Tra(true_y, true_u, t)

print("true_y.shape: ", true_y.shape)
print("true_u.shape: ", true_u.shape)
print(np.__version__)


def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s] # (M, D)
    batch_u0 = true_u[s]
    # batch_t = torch.stack([t[s + i] for i in range(args.batch_time)], dim=0).T
    batch_t = t[:args.batch_time]
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, 3)
    batch_u = torch.stack([true_u[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, 2)

    return batch_y0.to(device), batch_t.to(device), batch_y.to(device), batch_u0.to(device), batch_u.to(device)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y: torch.Tensor):
        # y[:,:,0:2] = y[:,:,0:2]**3
        return self.net(y**3)
    
if __name__ == '__main__':

    writer = init_writer()
    best_loss = float('inf')

    if args.viz:
        fig = plt.figure(figsize=(12, 4), facecolor='white')
        ax_traj = fig.add_subplot(121, frameon=False)
        ax_phase = fig.add_subplot(122, frameon=False)
        plt.show(block=False)

    ii = 0

    func = ODEFunc().to(device)
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y, batch_u0, batch_u = get_batch()
        
        # print("batch_y0.shape: ", batch_y0.shape)
        # print("batch_t.shape: ", batch_t.shape)
        # print("batch_y.shape: ", batch_y.shape)
        # print("batch_u0.shape: ", batch_u0.shape)
        # print("batch_u.shape: ", batch_u.shape)

        batch_init = torch.cat((batch_y0, batch_u0), dim=2)
        batch_data = torch.cat((batch_y, batch_u), dim=3)
        # print("batch_init.shape: ", batch_init.shape)
        # print("batch_data.shape: ", batch_data.shape)
        
        pred_y_aug = odeint(func, batch_init, batch_t).to(device)
        # print("pred_y.shape: ", pred_y.shape)
        loss = torch.mean(torch.abs(pred_y_aug - batch_data))
        # loss = torch.mean(torch.abs(pred_y_aug[:,:,:,0:2] - batch_y))
        loss.backward()
        optimizer.step()

        writer.add_scalar("training_loss", loss.item(), itr)


        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y_aug = odeint(func, torch.cat((true_y0,true_u0),dim =1), t).to(device)
                true_y_aug = torch.cat((true_y, true_u), dim=true_y.dim()-1)
                loss = torch.mean(torch.abs(pred_y_aug - true_y_aug))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                writer.add_scalar("Validation_loss", loss.item(), itr)
                if args.viz:
                    Plot_XY_Tra(true_y, pred_y_aug, t, ii, ax_traj, ax_phase)
                ii += 1

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    state = {'iteration': iter,
                        'model_state_dict': func.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}
                    
                    torch.save(state, "models/ode_input_aug.pth")

        end = time.time()


# def forward(self, t, y: torch.Tensor, u: torch.Tensor, Dt):
#         ii = int(t // Dt)
#         u_t_interpolated = true_u[ii] + (t - ii * Dt) * (true_u[ii + 1] - true_u[ii]) / Dt
#         return self.net(torch.cat((y**3, u), dim=2))