import sys  # nopep8
sys.path.append('.')  # nopep8

import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from core.lie_neurons_layers import *

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([[2., 0.]]).to(device)
t = torch.linspace(0., 25., args.data_size).to(device)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)


class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.mm(y**3, true_A)


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')


def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, 3)

    # batch_y0 = torch.cat((batch_y0,torch.zeros(batch_y0.shape[0],batch_y0.shape[1],1).to(device)),dim=2)
    # batch_y = torch.cat((batch_y,torch.zeros(batch_y.shape[0],batch_y.shape[1],batch_y.shape[2],1).to(device)),dim=3)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png_ln_ode_demo')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:
        # print("true_y", true_y.shape)
        # print("pred_y", pred_y.shape)
        # print("t", t.shape)
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        # print("hello",np.stack([x, y], -1).shape)
        dydt = odefunc(0, rearrange(torch.Tensor(np.stack([x, y, np.zeros_like(y)], -1).reshape(21 * 21, 3)),'b d -> b 1 d').to(device)).cpu().detach().numpy()
        # print("dydt",dydt.shape)
        dydt = rearrange(dydt,'b 1 d -> b d')
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 3)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('png_ln_ode_demo/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class LNODEFunc(nn.Module):

    def __init__(self):
        super(LNODEFunc, self).__init__()

        self.net = nn.Sequential(
            # LNLinearAndKillingRelu(1,50,algebra_type='so3'),
            # # LNLinear(50,1)
            # LNLinearAndKillingRelu(50,1,algebra_type='so3')

            
            LNLinearAndLieBracket(1,50,algebra_type='so3'),
            LNLinearAndLieBracket(50,1,algebra_type='so3')
            # LNLinearAndKillingRelu(1,50,algebra_type='so3'),
            # LNLinearAndLieBracket(50,50,algebra_type='so3'),
            # LNLinearAndKillingRelu(50,1,algebra_type='so3')
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                # nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        # print(y.shape)
        y3 = rearrange(y**3,'b c d -> b 1 d c')
        return rearrange(self.net(y3),'b 1 d c -> b c d')

class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 50),
            nn.Tanh(),
            nn.Linear(50, 3),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        # print(y.shape)
        return self.net(y**3)
    
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':

    ii = 0
    # print("true_y", true_y.shape)
    true_y = torch.cat((true_y,torch.zeros(true_y.shape[0],true_y.shape[1],1).to(device)),dim=2)
    true_y0 = torch.cat((true_y0,torch.zeros(true_y0.shape[0],1).to(device)),dim=1)
    
    true_y0 = rearrange(true_y0,'b d -> b 1 d')

    # func = LNODEFunc().to(device)
    func = ODEFunc().to(device)
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        # print("here")
        # print("batch_y", batch_y.shape)
        # print("batch_y0", batch_y0.shape)
        # print("batch_t", batch_t.shape)
        pred_y = odeint(func, batch_y0, batch_t).to(device)
        # print("pred y", pred_y.shape)
        loss = torch.mean(torch.abs(pred_y[:,:2] - batch_y[:,:2]))
        loss.backward()
        
        # for i in range(pred_y.shape[0]):
        #     print("loss:", loss.item())
        #     print("pred y", pred_y[i,:])
        #     print("true y", batch_y[i,:])
        #     print("----------")
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                # print("ivs")
                # print("true_y", true_y.shape)
                # print("true_y0", true_y0.shape)
                # print("t",t.shape)
                pred_y = odeint(func, true_y0, t)
                # print("pred_y", pred_y.shape)
                # print(pred_y)
                pred_y = pred_y[:,0,:,:]
                # print("pred_y", pred_y.shape)
                # for i in range(pred_y.shape[0]):
                #     print("pred y", pred_y[i,:])
                #     print("true y", true_y[i,:])
                #     print("----------")
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii)
                ii += 1

        end = time.time()
