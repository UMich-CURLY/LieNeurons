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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from core.lie_neurons_layers import *
from core.lie_alg_util import *
from core.lie_group_util import exp_so3
from experiment.euler_poincare_eq_layers import *

import emlp.nn.pytorch as emlpnn
from emlp.reps import T,V
from emlp.groups import SO

parser = argparse.ArgumentParser('Euler Poincare Equation Fitting')
parser.add_argument('--method', type=str, default='dopri5')
parser.add_argument('--num_training', type=int, default=10)
parser.add_argument('--num_testing', type=int, default=10)
parser.add_argument('--num_testing_augmentation', type=int, default=10)
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
# parser.add_argument('--training_config', type=str,
#                         default=os.path.dirname(os.path.abspath(__file__))+'/../config/euler_poincare/training_param.yaml')
parser.add_argument('--log_writer_path', type=str, default='logs/euler_poincare_ln')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

def init_writer():
    writer = SummaryWriter(
        args.log_writer_path+"_"+str(time.localtime()), args.model_type)
    writer.add_text("num_iterations: ", str(args.niters))

    return writer

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


torch.manual_seed(5566)
if args.num_training >1:
    training_y0 = torch.rand((args.num_training, 3)).to(device)
else:
    training_y0 = torch.tensor([[2., 1.,3.0]]).to(device)
val_true_y0 = torch.tensor([[2., 1.,3.0]]).to(device)
t = torch.linspace(0., 25., args.data_size).to(device)
t_val = torch.linspace(0., 5., int(args.data_size/5)).to(device)

# test data
torch.manual_seed(81292)
if args.num_testing > 1:
    testing_y0 = torch.rand((args.num_testing, 3)).to(device)
elif args.num_testing == 1:
    testing_y0 = torch.tensor([[2., 1.,3.0]]).to(device)

vis_y0 = torch.tensor([[2., 1.,3.0]]).to(device)

t_end_list = [5., 10., 15., 20., 25.]
t_test = []
for t_end in t_end_list:
    t_test.append(torch.linspace(0., t_end, int(args.data_size/25.*t_end)).to(device))
# vis_t = torch.linspace(0., args.viz_time, args.data_size).to(device)

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


class EMLPFunc(nn.Module):
    def __init__(self):
        super(EMLPFunc, self).__init__()

        G = SO(3)
        rep_in = V(G)
        rep_out = V(G)
        self.net = emlpnn.EMLP(rep_in,rep_out,group=G,num_layers=3,ch=128).to(device)
        self.R = torch.eye(3)

    def set_R(self,R):
        self.R = R.to(self.R.device)

    def forward(self, t, y):
        # y = y.squeeze(1)
        return self.net(y)
    
class EMLPFunc2(nn.Module):
    def __init__(self):
        super(EMLPFunc2, self).__init__()

        G = SO(3)
        rep_in = 4*V(G)
        rep_out = V(G)
        self.m = nn.Parameter(torch.randn(1,9))
        self.net = emlpnn.EMLP(rep_in,rep_out,group=G,num_layers=3,ch=128).to(device)
        self.R = torch.eye(3).to(device)

    def set_R(self,R):
        self.R = R.to(self.R.device)

    def forward(self, t, y):
        B = y.shape[0]
        # y = y.unsqueeze(1)
        # print("y",y.shape)
        # print("m",self.m.shape)
        m1 = self.m.reshape(3,3)
        m1 = self.R @ m1
        m1 = m1.reshape(1,9) 
        y = torch.cat([y,m1.unsqueeze(1).repeat(B,1,1)],2)
        return self.net(y)
    

with torch.no_grad():
    training_y = []
    for i in range(args.num_training):
        true_y = odeint(EulerPoincareEquation(), training_y0[i,:].unsqueeze(0), t, method='dopri5')
        training_y.append(true_y)   

    val_true_y = odeint(EulerPoincareEquation(), val_true_y0, t_val, method='dopri5')

''' 
N: number of testing data
J: number of end time being tested (5)
M_j: data size for j-th time
D: dimension (3)
'''
with torch.no_grad():
    testing_y = []
    # print("testing y0", testing_y0.shape)
    for j, t_j in enumerate(t_test):
        # for i in range(args.num_testing):
        true_y = odeint(EulerPoincareEquation(), testing_y0, t_j, method='dopri5').unsqueeze(-2)    # (M_j, N, 1, D)
        # print("true_y", true_y.shape)
        testing_y.append(true_y) # (J, (M_j, N, 1, D))

    # vis_y = odeint(EulerPoincareEquation(), vis_y0, vis_t, method='dopri5')

def test(func):
    for j, t_j in enumerate(t_test):
        with torch.no_grad():
            func.eval()
            pred_y = odeint(func, testing_y0.unsqueeze(1), t_j).to(device)  # Lie Neuron takes additional feature dimension thus we need to unsqueeze(1)
            cur_y = testing_y[j]    # 
            loss = torch.mean(torch.norm(pred_y - cur_y, dim=-1))

            time_meter.update(time.time() - end)
            print("loss at",t_end_list[j],"sec is", loss.item())

            # generate data for change of reference frame testing
            hat_so3 = HatLayer(algebra_type='so3')
            v = torch.rand((args.num_testing_augmentation,3))
            v = torch.div(v,torch.norm(v,dim=-1).unsqueeze(-1))
            phi = (math.pi-1e-6)*torch.rand(args.num_testing_augmentation,1)
            v = phi*v
            v_hat = hat_so3(v)
            R = exp_so3(v_hat).to(device)

    for j, t_j in enumerate(t_test):
        with torch.no_grad():
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


def get_training_batch():
    j = random.randint(0, args.num_training - 1)
    y_j = training_y[j]
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = y_j[s] # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([y_j[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, 3)

    return batch_y0, batch_t, batch_y

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs(args.fig_save_path)
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
        ax_traj.plot(t_val.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t_val.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t_val.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t_val.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t_val.cpu().min(), t_val.cpu().max())
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
        plt.savefig(args.fig_save_path+'/{:03d}'.format(itr))
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

    def get_avg(self):
        return self.avg

if __name__ == '__main__':

    ii = 0
    jj = 0

    writer = init_writer()


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
    elif args.model_type == 'neural_ode2':
        func = ODEFunc2().to(device)
    elif args.model_type == 'neural_ode3':
        func = ODEFunc3().to(device)
    elif args.model_type == 'emlp':
        func = EMLPFunc().to(device)
    elif args.model_type == 'emlp2':
        func = EMLPFunc2().to(device)
    
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)
    
    best_loss = float('inf')

    for itr in range(1, args.niters + 1):
        func.train()
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_training_batch()
        # print("here")
        # print("batch_y", batch_y.shape)
        # print("batch_y0", batch_y0.shape)
        # print("batch_t", batch_t.shape)
        pred_y = odeint(func, batch_y0, batch_t, atol=1e-9,rtol=1e-7).to(device)
        # print("pred y", pred_y.shape)
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()

        writer.add_scalar('training loss', loss.item(), itr)
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
                pred_y = odeint(func, val_true_y0.unsqueeze(0), t_val)
                # print("pred_y", pred_y.shape)
                # print(pred_y)
                pred_y = pred_y[:,0,:,:]
                # print("pred_y", pred_y.shape)
                # for i in range(pred_y.shape[0]):
                #     print("pred y", pred_y[i,:])
                #     print("true y", true_y[i,:])
                #     print("----------")
                # print(torch.abs(pred_y - val_true_y).shape)
                # print(torch.mean(torch.abs(pred_y - val_true_y)))
                loss = torch.mean(torch.abs(pred_y - val_true_y))

                writer.add_scalar('validation loss', loss.item(), itr)
                
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(val_true_y, pred_y, func, ii)
                ii += 1

            if loss < best_loss:
                best_loss = loss

                state = {'iteration': iter,
                        'model_state_dict': func.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}

                torch.save(state,  args.model_save_path +
                        '_best_val_loss_acc.pt')

                test(func)
                func.set_R(torch.eye(3))
                
            print("------------------------------")

        if itr % args.save_freq == 0:
            state = {'iteration': iter,
                        'model_state_dict': func.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss}

            torch.save(state, args.model_save_path + '_iter_'+str(itr)+'.pt')
            
        end = time.time()
