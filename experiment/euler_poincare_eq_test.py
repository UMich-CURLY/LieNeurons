import sys  # nopep8
sys.path.append('.')  # nopep8

import os
import argparse
import time
import numpy as np
import yaml
from tqdm import tqdm
import time
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from core.lie_neurons_layers import *
from core.lie_alg_util import *
from core.lie_group_util import *
from experiment.euler_poincare_eq_layers import *


parser = argparse.ArgumentParser('Euler Poincare Equation Fitting')
parser.add_argument('--method', type=str, default='dopri5')
parser.add_argument('--no_quantitative', action='store_true')
parser.add_argument('--no_test_augmentation', action='store_true')
parser.add_argument('--num_testing', type=int, default=10)
parser.add_argument('--num_testing_augmentation', type=int, default=10)
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--save_freq', type=int, default=100)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--viz_time', type=int, default=15)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--inertia_type', type=str, choices=['iss','model1'], default='iss')
parser.add_argument('--fig_save_path', type=str, default='figures/euler_poincare_ln')
parser.add_argument('--model_load_path', type=str, default='weights/ode4_new_training_best_test_loss_acc.pt')
parser.add_argument('--model_type', type=str, default='neural_ode')
parser.add_argument('--log_writer_path', type=str, default='logs/euler_poincare_ln')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


torch.manual_seed(81292)
if args.num_testing > 1:
    testing_y0 = torch.rand((args.num_testing, 3)).to(device)
elif args.num_testing == 1:
    testing_y0 = torch.tensor([[2., 1.,3.0]]).to(device)

vis_y0 = torch.tensor([[2., 1.,3.0]]).to(device)

t_end_list = [5., 10., 15., 20., 25.]
t = []
for t_end in t_end_list:
    t.append(torch.linspace(0., t_end, int(args.data_size/25.*t_end)).to(device))
vis_t = torch.linspace(0., args.viz_time, args.data_size).to(device)

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


''' 
N: number of testing data
J: number of end time being tested (5)
M_j: data size for j-th time
D: dimension (3)
'''
with torch.no_grad():
    testing_y = []
    # print("testing y0", testing_y0.shape)
    for j, t_j in enumerate(t):
        # for i in range(args.num_testing):
        true_y = odeint(EulerPoincareEquation(), testing_y0, t_j, method='dopri5').unsqueeze(-2)    # (M_j, N, 1, D)
        # print("true_y", true_y.shape)
        testing_y.append(true_y) # (J, (M_j, N, 1, D))

    vis_y = odeint(EulerPoincareEquation(), vis_y0, vis_t, method='dopri5')
    


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


# if args.viz:
    # makedirs(args.fig_save_path)
    # fig = plt.figure(figsize=(12, 4), facecolor='white')
    # ax_traj = fig.add_subplot(131, frameon=False)
    # ax_phase = fig.add_subplot(132, frameon=False)
    # ax_vecfield = fig.add_subplot(133, frameon=False)
    # plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):
    makedirs(args.fig_save_path)
    # plt.rcParams['text.usetex'] = True

    plt.rcParams.update({'font.size': 22})
    fig1 = plt.figure(1)
    # fig1.cla()
    
    plt.plot(vis_t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], '-', color='royalblue', linewidth=2)
    plt.plot(vis_t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', color='orange', linewidth=2)
    plt.xlim(vis_t.cpu().min(), vis_t.cpu().max())
    plt.title('Trajectories (x)')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.legend(['Ground Truth', 'Lie Neurons (x)'])
    # ax_traj.set_ylim(-2, 2)
    # fig1.legend()

    fig2 = plt.figure(2)
    plt.plot(vis_t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], '-',color='royalblue', linewidth=2)
    plt.plot(vis_t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], '--', color='orange', linewidth=2)
    plt.xlim(vis_t.cpu().min(), vis_t.cpu().max())
    plt.title('Trajectories (y)')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend(['True y', 'Predicted y'])

    fig3 = plt.figure(3)
    plt.plot(vis_t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 2], '-', color='royalblue', linewidth=2)
    plt.plot(vis_t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 2], '--',color='orange', linewidth=2)
    plt.xlim(vis_t.cpu().min(), vis_t.cpu().max())
    plt.title('Trajectories (z)')
    plt.xlabel('t')
    plt.ylabel('z')
    plt.legend(['True z', 'Predicted z'])

    fig4 = plt.figure(4)
    plt.plot(vis_t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], '-', color='royalblue', linewidth=2)
    plt.plot(vis_t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], '-', color='darkred', linewidth=2)
    plt.plot(vis_t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 2], '-', color='darkviolet', linewidth=2)
    plt.plot(vis_t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--',color='orange', linewidth=2)
    plt.plot(vis_t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], '--',color='mediumseagreen', linewidth=2)
    plt.plot(vis_t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 2], '--',color='gold', linewidth=2)
    plt.xlim(vis_t.cpu().min(), vis_t.cpu().max())
    plt.title('Trajectories')
    plt.xlabel('t')
    plt.ylabel('x, y, z')
    plt.legend(['Ground Truth x', 'Ground Truth y', 'Ground Truth z', 'Lie Neurons (No Mixing) x', 'Lie Neurons (No Mixing) y', 'Lie Neurons (No Mixing) z'])

    ax = plt.figure(5).add_subplot(projection='3d')
    ax.plot(true_y.cpu().numpy()[:, 0, 0],true_y.cpu().numpy()[:, 0, 1],true_y.cpu().numpy()[:, 0, 2], '-', color='royalblue')
    ax.plot(pred_y.cpu().numpy()[:, 0, 0],pred_y.cpu().numpy()[:, 0, 1],pred_y.cpu().numpy()[:, 0, 2], '-', color='orange')
    plt.title('Trajectories (3D)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend(['Ground Truth', 'Lie Neurons (No Mixing)'])
    # fig = plt.figure(figsize=(12, 4), facecolor='white')
    # ax_traj = fig.add_subplot(131, frameon=False)
    # ax_phase = fig.add_subplot(132, frameon=False)
    # ax_vecfield = fig.add_subplot(133, frameon=False)
    # plt.show(block=False)

    # ax_phase.cla()
    # ax_phase.set_title('Phase Portrait')
    # ax_phase.set_xlabel('x')
    # ax_phase.set_ylabel('y')
    # ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
    # ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
    # ax_phase.set_xlim(-2, 2)
    # ax_phase.set_ylim(-2, 2)

    # ax_vecfield.cla()
    fig6 = plt.figure(6)
    plt.title('Learned Vector Field')
    plt.xlabel('x')
    plt.ylabel('y')

    y, x = np.mgrid[-2:2:21j, -2:2:21j]
    # print("hello",np.stack([x, y], -1).shape)
    dydt = odefunc(0, rearrange(torch.Tensor(np.stack([x, y, np.zeros_like(y)], -1).reshape(21 * 21, 3)),'b d -> b 1 d').to(device)).cpu().detach().numpy()
    # print("dydt",dydt.shape)
    dydt = rearrange(dydt,'b 1 d -> b d')
    mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
    dydt = (dydt / mag)
    dydt = dydt.reshape(21, 21, 3)

    plt.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")

    # fig.tight_layout()
    plt.savefig(args.fig_save_path+'/{:03d}'.format(itr))
    plt.show()
    plt.draw()
    plt.waitforbuttonpress()


    
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

    def get_avg(self):
        return self.avg


if __name__ == '__main__':

    ii = 0
    jj = 0

    # load yaml file
    # config = yaml.safe_load(open(args.training_config))

    # writer = init_writer(config)

    
    # true_y0 = rearrange(true_y0,'b d -> b 1 d')

    if args.model_type == 'LN_ode':
        func = LNODEFunc(device=device).to(device)
    elif args.model_type == 'LN_ode2':
        func = LNODEFunc2(device=device).to(device)
    elif args.model_type == 'LN_ode3':
        func = LNODEFunc3(device=device).to(device)
    elif args.model_type == 'LN_ode4':
        func = LNODEFunc4(device=device).to(device)
    elif args.model_type == 'LN_ode5':
        func = LNODEFunc5(device=device).to(device)
    elif args.model_type == 'LN_ode6':
        func = LNODEFunc6(device=device).to(device)
    elif args.model_type == 'LN_ode7':
        func = LNODEFunc7(device=device).to(device)
    elif args.model_type == 'LN_ode8':
        func = LNODEFunc8(device=device).to(device)
    elif args.model_type == 'neural_ode':
        func = ODEFunc().to(device)

    checkpoint = torch.load(args.model_load_path)
    func.load_state_dict(checkpoint['model_state_dict'],strict=True)

    end = time.time()

    time_meter = RunningAverageMeter(1.00)
    

    ''' 
    N: number of testing data
    J: number of end time being tested (5)
    M_j: data size for j-th time
    D: dimension (3)
    testing_y0: (N, D) for each time j we start from the same y0
    testing_y: (J, (N, M_j, D))
    '''
    if not args.no_quantitative:
        for j, t_j in enumerate(t):
            with torch.no_grad():
                pred_y = odeint(func, testing_y0.unsqueeze(1), t_j).to(device)  # Lie Neuron takes additional feature dimension thus we need to unsqueeze(1)
                cur_y = testing_y[j]    # 
                loss = torch.mean(torch.norm(pred_y - cur_y, dim=-1))

                time_meter.update(time.time() - end)
                print("loss at",t_end_list[j],"sec is", loss.item())

    if not args.no_test_augmentation:
        # generate data for change of reference frame testing
        hat_so3 = HatLayer(algebra_type='so3')
        v = torch.rand((args.num_testing_augmentation,3))
        v = torch.div(v,torch.norm(v,dim=-1).unsqueeze(-1))
        phi = (math.pi-1e-6)*torch.rand(args.num_testing_augmentation,1)
        v = phi*v
        v_hat = hat_so3(v)
        R = exp_so3(v_hat).to(device)

        for j, t_j in enumerate(t):
            cur_y = testing_y[j]    # (N, M_j, D)
            loss_meter = RunningAverageMeter(1.00)
            for k in range(args.num_testing_augmentation):
                cur_R = R[k,:,:]
                # print("cur_R",cur_R.shape)
                # print("cur_y",cur_y.shape)
                # print("testing_y0",testing_y0.shape)
                cur_y0 = torch.einsum('ij,kj->ki',cur_R,testing_y0).unsqueeze(1)
                cur_y_conj = torch.einsum('ij,bksj->bksi',cur_R,cur_y)
                func.set_R(cur_R)
                with torch.no_grad():
                    pred_y = odeint(func, cur_y0, t_j).to(device)
                    loss = torch.mean(torch.norm(pred_y - cur_y_conj, dim=-1))
                    loss_meter.update(loss.item())

            print("conjugated loss at",t_end_list[j],"sec is", loss_meter.get_avg())

    if args.viz:
        with torch.no_grad():
            func.set_R(torch.eye(3).to(device))
            vis_pred_y = odeint(func, vis_y0.unsqueeze(0), vis_t).squeeze(1).to(device)
            print(vis_pred_y.shape)
            visualize(vis_y, vis_pred_y, func, ii)
            ii += 1
        
    
    # visualize(val_true_y, pred_y, func, ii)
    # ii += 1

            
    end = time.time()
