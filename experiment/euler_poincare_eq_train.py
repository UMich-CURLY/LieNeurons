import sys  # nopep8
sys.path.append('.')  # nopep8

import os
import argparse
import time
import numpy as np
import yaml
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from core.lie_neurons_layers import *
from core.lie_alg_util import *
from experiment.euler_poincare_eq_layers import *


def init_writer(config):
    writer = SummaryWriter(
        config['log_writer_path']+"_"+str(time.localtime()), comment=config['model_description'])
    writer.add_text("train_data_path: ", config['train_data_path'])
    writer.add_text("model_save_path: ", config['model_save_path'])
    writer.add_text("log_writer_path: ", config['log_writer_path'])
    writer.add_text("shuffle: ", str(config['shuffle']))
    writer.add_text("batch_size: ", str(config['batch_size']))
    writer.add_text("init_lr: ", str(config['initial_learning_rate']))
    writer.add_text("num_epochs: ", str(config['num_epochs']))

    return writer

parser = argparse.ArgumentParser('Euler Poincare Equation Fitting')
parser.add_argument('--method', type=str, default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--save_freq', type=int, default=100)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--fig_save_path', type=str, default='euler_poincare_ln')
parser.add_argument('--model_save_path', type=str, default='euler_poincare_ln')
parser.add_argument('--model_type', type=str, default='neural_ode')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([[2., 1.,3.0]]).to(device)
t = torch.linspace(0., 25., args.data_size).to(device)

class EulerPoincareEquation(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        '''
        Inertia matrix of the ISS
        https://athena.ecs.csus.edu/~grandajj/ME296M/space.pdf
        page 7-62
        '''
        # self.I = torch.Tensor([[12, 0., 0.],[0., 20., 0.],[0., 0., 5.]]).unsqueeze(0).to(device)
        self.I = torch.Tensor([[12, -5., 7.],[-5., 20., -2.],[7., -2., 5.]]).unsqueeze(0).to(device)
        # self.I = torch.Tensor([[12, 0, 0],[0, 20., 0],[0, 0, 5.]]).unsqueeze(0).to(device)
        # self.I = torch.Tensor([[5410880., -246595., 2967671.],[-246595., 29457838., -47804.],[2967671., -47804., 26744180.]]).unsqueeze(0).to(device)
        self.I_inv = torch.inverse(self.I)
        self.hat_layer = HatLayer(algebra_type='so3').to(device)

    def forward(self,t,w):
        '''
        w: angular velocity (B,3) or (1,3)
        '''
        # print("I",self.I.shape)
        # print("w hat",self.hat_layer(w).shape)
        # print("I inv",self.I_inv.shape)
        # print("w",w.shape)
        # print("w",w)
        w_v = w.unsqueeze(2)
        # print("return",(-torch.matmul(self.I_inv,torch.matmul(self.hat_layer(w),torch.matmul(self.I,w_v)))).shape)
        # print("----------------")
        # print("w_v",w_v.shape)
        # print("Iw",torch.matmul(self.I,w_v).shape)
        # print("hat i w",torch.matmul(self.hat_layer(w),torch.matmul(self.I,w_v)).shape)
        # print("I inv hat i w",torch.matmul(self.I_inv,torch.matmul(self.hat_layer(w),torch.matmul(self.I,w_v))).shape)
        return -torch.matmul(self.I_inv,torch.matmul(self.hat_layer(w),torch.matmul(self.I,w_v))).squeeze(2)


with torch.no_grad():
    true_y = odeint(EulerPoincareEquation(), true_y0, t, method='dopri5')


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
    makedirs('figures/'+args.fig_save_path)
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
        # ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        # ax_phase.set_xlim(-2, 2)
        # ax_phase.set_ylim(-2, 2)

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
        # ax_vecfield.set_xlim(-2, 2)
        # ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('figures/'+args.fig_save_path+'/{:03d}'.format(itr))
        # plt.draw()
        # plt.pause(0.001)



    
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
    jj = 0
    # print("true_y", true_y.shape)
    # true_y = torch.cat((true_y,torch.zeros(true_y.shape[0],true_y.shape[1],1).to(device)),dim=2)
    # true_y0 = torch.cat((true_y0,torch.zeros(true_y0.shape[0],1).to(device)),dim=1)
    
    true_y0 = rearrange(true_y0,'b d -> b 1 d')

    if args.model_type == 'LN_ode':
        func = LNODEFunc(device=device).to(device)
    elif args.model_type == 'LN_ode2':
        func = LNODEFunc2(device=device).to(device)
    elif args.model_type == 'LN_ode3':
        func = LNODEFunc3(device=device).to(device)
    elif args.model_type == 'LN_ode4':
        func = LNODEFunc4(device=device).to(device)
    elif args.model_type == 'neural_ode':
        func = ODEFunc().to(device)
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    # optimizer = optim.Adam(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)
    
    best_loss = float('inf')

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        # print("here")
        # print("batch_y", batch_y.shape)
        # print("batch_y0", batch_y0.shape)
        # print("batch_t", batch_t.shape)
        pred_y = odeint(func, batch_y0, batch_t, atol=1e-9,rtol=1e-7).to(device)
        # print("pred y", pred_y.shape)
        loss = torch.mean(torch.abs(pred_y[:,:] - batch_y[:,:]))
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

            if loss < best_loss:
                best_loss = loss

                state = {'iteration': iter,
                        'model_state_dict': func.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}

                torch.save(state, 'weights/' + args.model_save_path +
                        '_best_test_loss_acc.pt')
            print("------------------------------")

        if itr % args.save_freq == 0:
            state = {'iteration': iter,
                        'model_state_dict': func.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}

            torch.save(state, 'weights/'+ args.model_save_path + '_iter_'+str(itr)+'.pt')
            
        end = time.time()
